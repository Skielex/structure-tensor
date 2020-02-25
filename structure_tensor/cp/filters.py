import cupy
import numpy
from cupyx.scipy.ndimage.filters import correlate
from scipy.ndimage.filters import _gaussian_kernel1d, _ni_support


def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0):
    """One-dimensional Gaussian filter.

    Based on SciPy implementation.

    Arguments:
        input: array_like
            The input array.
        sigma: scalar
            Standard deviation for Gaussian kernel.
        axis: int, optional
            The axis of input along which to calculate. Default is -1.
        order : int, optional
            An order of 0 corresponds to convolution with a Gaussian
            kernel. A positive order corresponds to convolution with
            that derivative of a Gaussian.
        output: cupy.ndarray, dtype or None
            The array in which to place the output.
        mode: str
            The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval: scalar
            Value to fill past edges of input if mode is ``constant``.
            Default is ``0.0``.
        truncate : float, optional
            Truncate the filter at this many standard deviations.
            Default is 4.0.

    Returns:
        gaussian_filter1d: cupy.ndarray
    """
    input = cupy.asarray(input)
    sd = float(sigma)
    # Make the radius of the filter equal to truncate standard deviations.
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel.
    shape = numpy.ones(len(input.shape), dtype=numpy.int)
    shape[axis] = -1
    # Create weights.
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1].reshape(tuple(shape))
    weights = cupy.asarray(weights)
    return correlate(input, weights, output, mode, cval, 0)


def gaussian_filter(input, sigma, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0):
    """Multidimensional Gaussian filter.

    Based on SciPy implementation.

    Arguments:
        input: array_like
            The input array.
        sigma: scalar or sequence of scalars
            Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a
            sequence, or as a single number, in which case it is equal for
            all axes.
        order: int or sequence of ints, optional
            The order of the filter along each axis is given as a sequence
            of integers, or as a single number.  An order of 0 corresponds
            to convolution with a Gaussian kernel. A positive order
            corresponds to convolution with that derivative of a Gaussian.
        output: cupy.ndarray or None
            The array in which to place the output.
        mode: str
            The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval: scalar
            Value to fill past edges of input if mode is ``constant``.
            Default is ``0.0``.
        truncate : float
            Truncate the filter at this many standard deviations.
            Default is 4.0.

    Returns:
        gaussian_filter: cupy.ndarray
            Returned array of same shape as `input`.

    Notes:
        The multidimensional filter is implemented as a sequence of
        one-dimensional convolution filters. The intermediate arrays are
        stored in the same data type as the output. Therefore, for output
        types with a limited precision, the results may be imprecise
        because intermediate results may be stored with insufficient
        precision.
    """
    input = cupy.array(input)
    if output is None:
        output = cupy.zeros_like(input)

    # Code from Scipy function.
    orders = _ni_support._normalize_sequence(order, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii]) for ii in range(len(axes)) if sigmas[ii] > 1e-15]

    if len(axes) > 0:

        # Create valiable for output.
        out = output
        for axis, sigma, order, mode in axes:

            # Flatten other axis.
            # This seems to improve performance at the cost of some memory.
            out_flat = cupy.moveaxis(out, axis, -1)
            out_shape = out_flat.shape
            out_flat = out_flat.reshape(-1, out_flat.shape[-1])
            input_flat = cupy.moveaxis(input, axis, -1)
            input_flat = input_flat.reshape(-1, input_flat.shape[-1])

            # Do 1D filtering.
            out = gaussian_filter1d(input_flat, sigma, -1, order, out_flat, mode, cval, truncate)
            out = cupy.moveaxis(out.reshape(out_shape), -1, axis)

            # Swap input and out. This was we only use two arrays.
            tmp = input
            input = out
            out = tmp

        # If there was an even number of iterations we need to copy values from input to output.
        if len(axes) % 2 == 0:
            output[:] = input
    else:
        output[...] = input[...]
    return output
