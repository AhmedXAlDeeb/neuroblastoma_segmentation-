#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import vtk
import logging
import argparse
import functools
import numpy as np

'''
This simple python script converts a STL mesh surface to Nifti image. The Output Volume Space
will match the one of reference nifti image that must be provided.
The conversion procedure use the vtk library.
'''

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


#
# Definition of loggin levels
#

log_levels = {
    0: logging.ERROR,
    1: logging.WARN,
    2: logging.INFO,
    3: logging.DEBUG,
}


#
# Simple CLI definition using argparse
#

def parse_args():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument('-in',
                            '--input',
                            dest='input',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to the input STL model. Must be .stl')
    _ = parser.add_argument('-ref',
                            '--reference',
                            dest='reference',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to the reference Nifti image. Must be .nii or .nii.gz')
    _ = parser.add_argument('-out',
                            '--output',
                            dest='output',
                            action='store',
                            type=str,
                            required=True,
                            help='Output filename. Must be .nii or .nii.gz')
    #
    # Options for the verbosity level
    #
    # The code for the selection of the logging level was taken from:
    # https://gist.github.com/willprice/352bb7cd40de33e73b84b93b9ab3d240
    #

    parser.add_argument("-v",
                        "--verbose",
                        dest="verbosity",
                        action="count",
                        required=False,
                        default=0,
                        help="Verbosity (between 1-4 occurrences with more leading to more "
                         "verbose logging). ERROR=0, WARN=2, INFO=3, "
                         "DEBUG=4")

    args = parser.parse_args()
    return args

#
# Now define a decorator to use for upadate the various vtk objects
#
def update(func):
    '''
    This decorator allows ot update your vtk pipeline
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        vtk_filter = func(*args, **kwargs)
        _ = vtk_filter.Update()

        return vtk_filter
    return wrapper


#
# Define Basic Reading and Writing functions
#


@update
def read_stl(filename):
    """STL reader
    """
    logging.debug('Reading stl file from: {}'.format(filename))
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)

    return reader


@update
def read_nifti(filename):
    """Nifti Reader
    """
    logging.debug('Reading Nifti Image from: {}'.format(filename))
    reader = vtk.vtkNIFTIImageReader()
    _ = reader.SetFileName(filename)

    return reader


@update
def write_nifti(filename, image, qform_matrix, qfac):
    '''Nifti Writer
    '''

    logging.debug('Writing the image to: {}'.format(filename))

    writer = vtk.vtkNIFTIImageWriter()
    _ = writer.SetFileName(filename)
    _ = writer.SetInputData(image)
    _ = writer.SetQFormMatrix(qform_matrix)
    _ = writer.SetQFac(qfac)

    return writer



#
# Function Useful to Init the Required Spatial Information from the reference image
#

def get_surface_origin(bounds, spacing):

    logging.debug('Computing the Physical Origin og the Image Corrseponding to \
                  the mesh surface, using bounds: {} and \
                  spacing: {}'.format(bounds, spacing))

    origin = [bounds[2*i] + (s / 2) for i,s in enumerate(spacing)]

    logging.debug('Computed Origin is: {}'.format(origin))

    return tuple(origin)



def get_surface_dimensions(bounds, spacing):

    logging.debug('Getting the Surface Dimensions using bounds: {} and Spacing:\
                  {}'.format(bounds, spacing))

    dimensions = [(bounds[2*i+1] - bounds[2*i]) // spacing[i] for i in range(len(spacing))]
    dimensions = tuple(map(lambda x : int(x), dimensions))

    logging.debug('Estimated Dimensions: {}'.format(dimensions))

    return dimensions


def get_origin_from_qform_matrix(QFormMatrix):
    '''
    Get the physical origin
    '''

    logging.debug("Estimating Reference Image Origin From QFormMatrix")

    offset = [QFormMatrix.GetElement(i, 3) for i in range(3)]
    sign = [QFormMatrix.GetElement(i, i) for i in range(3)]
    origin = tuple([s * o for s, o in zip(sign, offset)])

    logging.debug("Estimated Origin: {}".format(origin))

    return origin


def get_reference_information_from_image(reader, surface_bounds):
    ''' Get the Configuration Dictionary using the reference image
    '''

    logging.debug('Parsing the image get the required information')

    surface_dimensions = get_surface_dimensions(surface_bounds, reader.GetOutput().GetSpacing())
    surface_origin = get_surface_origin(surface_bounds, reader.GetOutput().GetSpacing())
    referece_origin = get_origin_from_qform_matrix(reader.GetQFormMatrix())

    config = {'Reference Dimensions' : reader.GetOutput().GetDimensions(),
              'Surface Dimensions' : surface_dimensions,
              'Spacing' : reader.GetOutput().GetSpacing(),
              'Reference Origin' : referece_origin,
              'Surface Origin' : surface_origin,
              'Direction' : reader.GetOutput().GetDirectionMatrix(),
              'QFormMatrix' : reader.GetQFormMatrix(),
              'QFac' : reader.GetQFac()}

    return config


#
# Function to convert the input mesh surface to the binary image volume
#

def init_vtk_image(spacing, dimensions, origin, direction, constant_value=1):
    '''
    Create a vtk image acording to the specified spacing, dimensions, origin and
    costant_value.
    '''
    logging.debug('Creating A VTK imaeg with Dimension: {}, Spacing: {}, \
                  Origin: {}, Direction: {}, GL: {}'.format(dimensions, spacing, origin, direction, constant_value))

    vtkImage = vtk.vtkImageData()
    _ = vtkImage.SetSpacing(spacing)
    _ = vtkImage.SetDimensions(dimensions)
    _ = vtkImage.SetDirectionMatrix(direction)
    _ = vtkImage.SetOrigin(origin)
    _ = vtkImage.AllocateScalars(7, 1) # 7 implise the usage of unsigned int type

    number_of_points = vtkImage.GetNumberOfPoints()

    for i in range(number_of_points):
        vtkImage.GetPointData().GetScalars().SetTuple1(i, 1)

    return vtkImage


@update
def vtk_polydata2imagestencil(polydata, origin, spacing, extent):
    '''
    Intialize vtkPolyDataToImageStencil.
    '''

    poly2stenc = vtk.vtkPolyDataToImageStencil()
    _ = poly2stenc.SetInputData(polydata)
    _ = poly2stenc.SetOutputOrigin(origin)
    _ = poly2stenc.SetOutputSpacing(spacing)
    _ = poly2stenc.SetOutputWholeExtent(extent)

    return poly2stenc


@update
def get_image_stencil(vtk_image, poly2stencil=None, bkg=0):
    '''
    '''
    image_stencil = vtk.vtkImageStencil()
    _ = image_stencil.SetInputData(vtk_image)

    if poly2stencil is not None:
        _ = image_stencil.SetStencilConnection(poly2stencil.GetOutputPort())
    _ = image_stencil.ReverseStencilOff()
    _ = image_stencil.SetBackgroundValue(bkg)

    return image_stencil


@update
def translate_image(image, offset=[0., 0., 0.], bkg_level=0):
    '''Translate the image to the reference origin
    '''

    logging.debug('Transalete the imgage of: {}'.format(offset))
    transform = vtk.vtkTransform()
    _ = transform.Translate(*offset)

    reslice = vtk.vtkImageReslice()
    _ = reslice.SetResliceTransform(transform)
    _ = reslice.SetInterpolationModeToNearestNeighbor()
    _ = reslice.SetInputData(image)
    _ = reslice.SetOutputSpacing(image.GetSpacing())
    _ = reslice.SetOutputOrigin(image.GetOrigin())
    _ = reslice.SetOutputExtent(image.GetExtent())
    _ = reslice.SetBackgroundLevel(bkg_level)

    return reslice

@update
def vtk_change_image_information(image, origin):
    '''
    '''

    changer = vtk.vtkImageChangeInformation()
    _ = changer.SetInputData(image)
    #_ = changer.SetInformationInputData(reference)
    _ = changer.SetOutputOrigin(origin)
    return changer


#
# Main Function Containing the Whole Pipeline
#

def main():

    args = parse_args()

    # set the logging level
    logging.basicConfig(level=log_levels[min(args.verbosity, max(log_levels.keys()))],
                        format='%(asctime)s - %(name)s -  %(levelname)s - %(message)s')

    logging.info('Reading input and reference image')

    surface = read_stl(args.input)
    reference = read_nifti(args.reference)

    logging.info('Converting Mesh to Binary Image')

    surface_bounds = surface.GetOutput().GetBounds()
    logging.debug('Extracted Surface Bounds: {}'.format(surface_bounds))

    config = get_reference_information_from_image(reference, surface_bounds)

    logging.debug('Extracted Reference Informations: {}'.format(config))

    logging.info("Starting the Creation of the Base Image")

    vtk_image = init_vtk_image(spacing=config['Spacing'],
                               dimensions=config['Reference Dimensions'],
                               origin=config['Reference Origin'],
                               direction=config['Direction'],
                               constant_value=1)


    logging.info('Converting PolyData to Image Stencil')
    poly2stencil = vtk_polydata2imagestencil(polydata=surface.GetOutput(),
                                            origin=config['Surface Origin'],
                                            spacing=config['Spacing'],
                                            extent=vtk_image.GetExtent())
    logging.info('Get Image Stencil')
    image_stencil = get_image_stencil(vtk_image=vtk_image,
                                      poly2stencil=poly2stencil,
                                      bkg=0)

    logging.info('Positiong the volume in the space')

    offset = np.asarray(config['Reference Origin']) - np.asarray(config['Surface Origin'])

    logging.debug('The computed offset is: {}'.format(offset))

    translated = translate_image(image_stencil.GetOutput(), offset)
    translated = vtk_change_image_information(translated.GetOutput(), (0., 0., 0.))

    logging.info('Writing the results')

    _ = write_nifti(filename=args.output,
                    image=translated.GetOutput(),
                    qform_matrix=config['QFormMatrix'],
                    qfac=config['QFac'])

    logging.info('DONE')



if __name__ == '__main__':
    main()