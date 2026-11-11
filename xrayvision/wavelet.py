"""
Compressed sensing image reconstruction with the finite isotropic wavelet transform (VIS_WV).
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from numpy.typing import NDArray

import astropy.units as apu
from astropy.units import Quantity

from sunpy.map import GenericMap, Map

from xrayvision.imaging import generate_header, validate_and_expand_kwarg, vis_psf_image, vis_to_image
from xrayvision.utils import get_logger
from xrayvision.visibility import Visibilities

__all__ = ["vis_wv"]

logger = get_logger(__name__, "DEBUG")


def _meyer_aux(x: NDArray[np.float64]) -> NDArray[np.float64]:
    y = 35 * x**4 - 84 * x**5 + 70 * x**6 - 20 * x**7
    return np.where(x > 1, 1.0, np.where(x >= 0, y, 0.0))


def _meyer_scaling(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.abs(omega)
    fall = np.cos(np.pi / 2 * _meyer_aux(4 * x / np.pi - 1))
    return np.where(x < np.pi / 4, 1.0, np.where(x <= np.pi / 2, fall, 0.0))


def _meyer_wavelet(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.abs(omega)
    rise = np.sin(np.pi / 2 * _meyer_aux(4 * x / np.pi - 1))
    fall = np.cos(np.pi / 2 * _meyer_aux(2 * x / np.pi - 1))
    return np.where(x <= np.pi / 4, 0.0, np.where(x <= np.pi / 2, rise, np.where(x <= np.pi, fall, 0.0)))


def _fiwt_spectra(n: int, nscales: int, pixel_size: float) -> NDArray[np.float64]:
    r"""
    Return the 2D isotropic Meyer spectra, scaling function first, then one wavelet per scale :math:`a_j = 2^{-j}`.

    The spectra are ``ifftshift`` ed so they can multiply unshifted FFTs directly; together they form a Parseval
    frame, :math:`\phi^2 + \sum_j \psi_j^2 = 1`.
    """
    # Omega fixes the bands in arcsec^-1 independent of the pixel size
    omega = 2.0 ** (nscales - 1) / pixel_size
    xi = np.linspace(-omega, omega, n)
    dist = ifftshift(np.hypot(*np.meshgrid(xi, xi)))
    psi = np.stack([_meyer_scaling(dist)] + [_meyer_wavelet(2.0**-j * dist) for j in range(nscales)])
    # SSW lets the finest band fall to 0 above ~pi/4 arcsec^-1, losing the frame for pixels below ~0.9 arcsec,
    # keep it flat at 1 instead; this only changes frequencies past its peak
    psi[-1] = np.sqrt(np.clip(1 - (psi[:-1] ** 2).sum(axis=0), 0, None))
    return psi


def _fiwt(image: NDArray[np.float64], psi: NDArray[np.float64]) -> NDArray[np.float64]:
    # psi is real and even so the coefficients of a real image are real
    return ifft2(psi * fft2(image)).real


def _ifiwt(coeffs: NDArray[np.float64], psi: NDArray[np.float64]) -> NDArray[np.float64]:
    return ifft2((psi * fft2(coeffs)).sum(axis=0)).real


@apu.quantity_input  # type: ignore[untyped-decorator]
def vis_wv(
    vis: Visibilities,
    shape: Quantity[apu.pix] = (129, 129) * apu.pix,
    pixel_size: Quantity[apu.arcsec / apu.pix] = (1, 1) * apu.arcsec / apu.pix,
    *,
    nscales: int = 3,
    lam: float | None = None,
    maxiter: int = 200,
    map: bool = True,  # noqa: A002
) -> GenericMap | Quantity:
    r"""
    Finite isotropic wavelet transform compressed sensing (VIS_WV) image reconstruction.

    Solves

    .. math::

        \min_f \| H \mathbf{F} f - v \|_2^2 + \lambda \| W f \|_1

    with FISTA in the image domain (as a deconvolution of the dirty map by the PSF), where :math:`W` is the finite
    isotropic wavelet transform built from the Meyer wavelet. The total flux is constrained to the real part of the
    visibility with the shortest baseline and the result is clipped to be positive. A port of ``vis_wv.pro`` in SSW.

    Parameters
    ----------
    vis :
        Input visibilities.
    shape :
        Image shape, must be square and odd.
    pixel_size :
        Pixel size, must be square.
    nscales :
        Number of wavelet scales.
    lam :
        Regularisation parameter, if `None` estimated from the first unregularised iteration (Miller criterion).
    maxiter :
        Maximum number of FISTA iterations.
    map :
        Return a sunpy map or bare array.

    Returns
    -------
    :
        Reconstructed image in units of the visibilities per arcsec squared.

    Notes
    -----
    ``vis_wv.pro`` centres the PSF one pixel off the FFT origin, so its images are shifted one pixel down in x and y;
    here the PSF is centred correctly. Its finest wavelet band also falls to zero above ~:math:`\pi/4` arcsec\
    :sup:`-1`, so for pixels below ~0.9 arcsec the transform is no longer a frame; here that band is kept flat.
    Otherwise the results agree, except when the PSF sum is close to zero. In that case, the PSF and dirty map
    share a normalization by the PSF peak to avoid amplifying cancellation error in their sums.

    References
    ----------
    * :cite:t:`DuvalPoo2018_vis_wv`
    """
    shape = validate_and_expand_kwarg(shape, "shape")
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    n, m = shape.to_value(apu.pix).astype(int)
    if n != m or n % 2 == 0:
        raise ValueError(f"shape must be square and odd, got {shape}.")
    if pixel_size[0] != pixel_size[1]:
        raise ValueError(f"pixel_size must be square, got {pixel_size}.")
    pixel = pixel_size[0].to_value(apu.arcsec / apu.pix)

    psi = _fiwt_spectra(n, nscales, pixel)
    dirty_map: NDArray[np.float64] = vis_to_image(vis, shape=shape, pixel_size=pixel_size).value
    psf: NDArray[np.float64] = vis_psf_image(vis, shape=shape, pixel_size=pixel_size).value

    # Flux constraint from the shortest baseline
    radius = np.hypot(vis.u, vis.v)
    dc = vis.visibilities[radius == radius.min()].real.max()
    unit = dc.unit
    dc = dc.value

    psf_sum = psf.sum()
    if abs(psf_sum) <= 1e-12 * np.abs(psf).sum():
        # Missing zero-frequency data can make the signed PSF sum vanish.
        # Use the same stable scale for the PSF and dirty map in that case.
        scale = np.max(np.abs(psf))
        psf = psf / scale
        dirty_map = dirty_map / scale
    else:
        psf = psf / psf_sum
        dirty_map *= dc / dirty_map.sum()
    psf_ft = fft2(ifftshift(psf))
    dirty_ft = fft2(dirty_map)
    lipschitz = 2 * np.max(np.abs(psf_ft) ** 2)

    autolam = lam is None
    lam = 0.0 if lam is None else lam
    x = dirty_map
    y = x
    t = 1.0
    old_total = 0.0
    for i in range(1, maxiter + 1):
        x_old, t_old = x, t

        # Gradient step then soft threshold the wavelet coefficients
        y = y - 2 / lipschitz * ifft2(np.conj(psf_ft) * (psf_ft * fft2(y) - dirty_ft)).real
        wy = _fiwt(y, psi)
        wy = np.sign(wy) * np.clip(np.abs(wy) - lam / lipschitz, 0, None)
        x = _ifiwt(wy, psi)
        x = x - (x.sum() - dc) / n**2

        t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
        y = x + (t_old - 1) / t * (x - x_old)

        residual = dirty_map - ifft2(psf_ft * fft2(x)).real
        likelihood = np.sum(residual**2)
        sparsity = np.abs(_fiwt(x, psi)).sum()

        if i == 1 and autolam:
            # Capped so lambda can't be large enough to give a null solution
            lam = min(likelihood / sparsity, np.abs(ifft2(np.conj(psf_ft) * dirty_ft)).max())

        total = likelihood + lam * sparsity
        if i % 25 == 0:
            logger.debug(f"Iter: {i}, Obj function: {total}, likelihood: {likelihood}")
        if i > 9 and old_total <= total:
            break
        old_total = total

    logger.debug(f"Lambda: {lam}")
    image = np.clip(x, 0, None) / pixel**2 * unit / apu.arcsec**2

    if map:
        header = generate_header(vis, shape=shape, pixel_size=pixel_size)
        header["bunit"] = image.unit.to_string("fits")
        return Map((image.value, header))
    return image
