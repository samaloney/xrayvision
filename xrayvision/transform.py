"""
Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT) related functions

There are two implementations one a standard DFT `dft` and IDFT `idft` in terms of pixel space, i.e.
the input has no positional information other than an arbitary 0 origin and a length. The second
takes inputs which have positional information `dft_map` and the inverse `idft_map`

"""

import numpy as np


def generate_xy(number_pixels, center=0.0, pixel_size=1.0):
    """
    Generate the x or y coordinates given the number of pixels, center and pixel size

    Parameters
    ----------
    number_pixels : `int`
        Number of pixels
    center : `float`
        Center coordinates
    pixel_size : `float`
        Size of pixel in physical units (e.g. arcsecs, meters)

    Returns
    -------
    `numpy.array`
        The generated x, y coordinates

    See Also
    --------
    generate_uv : Generates corresponding coordinates but un u, v space

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> x = generate_xy(33)

    """
    x = (np.arange(number_pixels) - number_pixels / 2 + 0.5) * pixel_size + center
    return x


def generate_uv(number_pixels, center=0.0, pixel_size=1.0):
    """
    Generate the u or v  coordinates given the number of pixels, center and pixel size

    Parameters
    ----------
    number_pixels : `int`
        Number of pixels
    center : `float`
        Center coordinates
    pixel_size : `float`
        Size of pixel in physical units (e.g. arcsecs, meters)

    Returns
    -------
    `numpy.array`
        The generated u, v coordinates

    See Also
    --------
    generate_xy : Generates corresponding coordinate but un x, y space

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> x = generate_uv(33)

    """
    x = (np.arange(number_pixels) - number_pixels / 2 + 0.5) * (1 / (pixel_size * number_pixels))
    if center != 0.0:
        x += 1/center
    return x


def dft_map(input_array, uv, center=(0.0, 0.0), pixel_size=(1.0, 1.0)):
    """
    Calculate the discrete Fourier transform of the array or image in terms of coordinates \
    returning a 1-D array of complex visibilities

    Parameters
    ----------
    input_array : `numpy.ndarray`
        Input array to be transformed

    uv : `numpy.array`
        Array of u, v coordinates where visibilities will be calculated

    center : `tuple` (x, y), optional
        Coordinates of the center of the map

    pixel_size : `tuple` (x_size, y_size), optional
        The size of a pixel in (x_size, y_size) format

    Returns
    -------
    `numpy.ndarray`
        The complex visibilities evaluated at the u, v coordinates given bu `uv`

    """
    m, n = input_array.shape
    size = m * n
    vis = np.zeros(uv.shape[1], dtype=complex)

    x = generate_xy(m, center[0], pixel_size[0])
    y = generate_xy(n, center[1], pixel_size[1])

    x, y = np.meshgrid(x, y)
    x = x.reshape(size)
    y = y.reshape(size)

    for i in range(uv.shape[1]):
        vis[i] = np.sum(
            input_array.reshape(size) * np.exp(
                -2j * np.pi * (uv[0, i] * x + uv[1, i] * y)))

    return vis


def idft_map(input_vis, shape, uv, center=(0.0, 0.0), pixel_size=(1.0, 1.0)):
    """
    Calculate the inverse discrete Fourier transform in terms of coordinates returning a 2-D real
    array or image

    Parameters
    ----------
    input_vis : array-like
        The input visibilities to use

    shape : (x,y)
        The shape of the array

    uv : array-like
        The u, v coordinates corresponding to the input visibilities in `input_visibilities`

    center: array-like
        Position of the center of the transformation. The center
        of the result image is (0,0) and the direction of the x axis is ->
        and the direction of the y axis is ^

    pixel_size: array-like
        The size of a pixel in (x_size, y_size) format

    Returns
    -------
    array-like
        The complex visibilities evaluated at the u, v coordinates

    """
    m, n = shape
    size = m * n

    x = generate_xy(m, center[0], pixel_size[0])
    y = generate_xy(n, center[1], pixel_size[1])

    x, y = np.meshgrid(x, y)
    x = x.reshape(size)
    y = y.reshape(size)

    im = np.zeros(size)

    for i in range(size):
        im[i] = (1 / input_vis.size) * \
            np.real(np.sum(input_vis * np.exp(2j * np.pi * (uv[0, :] * x[i] + uv[1, :] * y[i]))))

    return im.reshape(m, n)


# def dft(im, uv):
#     """
#     Discrete Fourier transform of the input array or image calculated at the given u, v coordinates
#
#     Loops over a list of u, v coordinates rather than looping over u and v separately
#
#     Parameters
#     ----------
#     im :  `numpy.ndarray`
#         Input array
#
#     uv : `numpy.ndarray`
#         Array of u, v coordinates where visibilities will be calculated
#
#     Returns
#     -------
#     vis : `numpy.ndarray`
#         The complex visibilities evaluated at the u, v coordinates given bu `uv`
#
#     """
#     m, n = im.shape
#     size = im.size
#     vis = np.zeros(size, dtype=complex)
#     xy = np.mgrid[0:m, 0:n].reshape(2, size)
#     for i in range(uv.shape[1]):
#         vis[i] = np.sum(
#             im.reshape(size) * np.exp(
#                 -2j * np.pi * (uv[0, i] * xy[0, :] / m + uv[1, i] * xy[1, :] / n)))
#
#     return vis


# def idft(vis, shape, uv):
#     """
#     Inverse discrete Fourier transform of the input array or image calculated at the given u, v \
#     coordinates
#
#     Loops over a list of x, y pixels rather than looping over x and y separately
#
#     Parameters
#     ----------
#     vis: `numpy.array`
#         The input visibilities to use
#
#     shape :  `tuple` (x, y)
#         Size of image to create
#
#     uv : `numpy.ndarray`
#         Array of u, v coordinates corresponding to the visibilities in `vis`
#
#     Returns
#     -------
#     `numpy.ndarray`
#         The inverse transform or back projection
#
#     """
#     m, n = shape
#     size = m * n
#     out = np.zeros(m * n)
#     xy = np.mgrid[0:m, 0:n].reshape(2, size)
#     for i in range(size):
#         out[i] = (1 / vis.size) * np.sum(
#             vis * np.exp(
#                 2j * np.pi * (uv[0, :] * xy[0, i] / m + uv[1, :] * xy[1, i] / n)))
#
#     return out.reshape(m, n)