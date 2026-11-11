"""VIS-CS reconstruction using a sparse, adaptive Gaussian dictionary.

Based on Felix, Bolzern & Battaglia (2017), ApJ 849, 10,
https://doi.org/10.3847/1538-4357/aa8d1e, and SolarSoft VIS_CS 2.0.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.special import erf

import astropy.units as u

from xrayvision.visibility import Visibilities

__all__ = ["vis_cs"]

_Array = npt.NDArray[Any]


@dataclass
class _Dictionary:
    # Native SSW coordinates: positions measured from the lower left of the FOV.
    x: _Array
    y: _Array
    sx: _Array
    sy: _Array
    phi: _Array
    amp: _Array
    a: _Array
    b: _Array
    c: _Array
    viv: _Array

    def take(self, indices: Any) -> "_Dictionary":
        return _Dictionary(**{key: value[indices] for key, value in vars(self).items()})


def _dictionary(
    x: _Array, y: _Array, sx: _Array, sy: _Array, phi: _Array, shape: tuple[int, int], pixel: _Array
) -> _Dictionary:
    """Construct normalized Gaussians and their continuous volume in the FOV."""
    x, y, sx, sy, phi = (np.asarray(t, dtype=float) for t in (x, y, sx, sy, phi))
    a = np.cos(phi) ** 2 / (2 * sx**2) + np.sin(phi) ** 2 / (2 * sy**2)
    b = -np.sin(2 * phi) / (4 * sx**2) + np.sin(2 * phi) / (4 * sy**2)
    c = np.sin(phi) ** 2 / (2 * sx**2) + np.cos(phi) ** 2 / (2 * sy**2)
    amp = 1 / (2 * np.pi * sx * sy)
    ny, nx = shape
    dx, dy = pixel
    # Preserve SSW 2.0 integration bounds and its cross-term sign convention.
    xlo = -0.5 + 0.5 * dx - x
    xhi = nx * dx - 0.5 + 0.5 * dx - x
    ylo = -0.5 + 0.5 * dy - y
    yhi = ny * dy - 0.5 + 0.5 * dy - y
    viv = np.empty(len(x))
    for i in range(len(x)):
        root = np.sqrt(a[i])

        def integrand(t: float, i: int = i, root: float = root) -> float:
            return float(
                amp[i]
                * np.sqrt(np.pi)
                / (2 * root)
                * np.exp((b[i] ** 2 / a[i] - c[i]) * t**2)
                * (erf(root * xhi[i] - b[i] * t / root) - erf(root * xlo[i] - b[i] * t / root))
            )

        viv[i] = quad(integrand, ylo[i], yhi[i], epsabs=1e-10, epsrel=1e-7)[0]
    if np.any(~np.isfinite(viv)) or np.any(viv <= np.finfo(float).tiny):
        raise ValueError("Gaussian dictionary has zero or nonfinite volume in the field of view.")
    return _Dictionary(x, y, sx, sy, phi, amp, a, b, c, viv)


def _random_dictionary(
    rng: np.random.Generator,
    count: int,
    shape: tuple[int, int],
    pixel: _Array,
    min_sigma: float,
    max_sigma: float,
    anisotropy: float,
) -> _Dictionary:
    ny, nx = shape
    dx, dy = pixel
    r = rng.random((count, 4))
    sx = min_sigma + r[:, 2] ** 2 * (max_sigma - min_sigma)
    sy = np.clip(sx + rng.standard_normal(count) * sx * 0.25, sx / anisotropy, sx * anisotropy)
    sy = np.clip(sy, min_sigma, max_sigma)
    return _dictionary(r[:, 0] * nx * dx, r[:, 1] * ny * dy, sx, sy, r[:, 3] * np.pi / 2, shape, pixel)


def _refine_dictionary(
    words: _Dictionary,
    rng: np.random.Generator,
    count: int,
    shape: tuple[int, int],
    pixel: _Array,
    min_sigma: float,
    max_sigma: float,
    anisotropy: float,
) -> _Dictionary:
    nwords = len(words.x)
    each = max(0, (count // 2 - nwords) // nwords)
    parts = [words]
    if each:
        old = words.take(np.tile(np.arange(nwords), each))
        r = rng.standard_normal((len(old.x), 5))
        sx = np.clip(old.sx + r[:, 2] * old.sx * 0.4, old.sy / anisotropy, old.sy * anisotropy)
        sy = np.clip(old.sy + r[:, 3] * old.sy * 0.4, old.sx / anisotropy, old.sx * anisotropy)
        parts.append(
            _dictionary(
                old.x + r[:, 0] * 2 * pixel[0],
                old.y + r[:, 1] * 2 * pixel[1],
                np.clip(sx, min_sigma, max_sigma),
                np.clip(sy, min_sigma, max_sigma),
                old.phi + r[:, 4] * np.pi / 8,
                shape,
                pixel,
            )
        )
    remaining = count - sum(len(part.x) for part in parts)
    if remaining:
        parts.append(_random_dictionary(rng, remaining, shape, pixel, min_sigma, max_sigma, anisotropy))
    return _Dictionary(**{key: np.concatenate([getattr(part, key) for part in parts]) for key in vars(words)})


def _fourier(
    dictionary: _Dictionary,
    uu: _Array,
    vv: _Array,
    shape: tuple[int, int],
    pixel: _Array,
    phase_center: _Array | tuple[float, float] = (0.0, 0.0),
) -> _Array:
    """Analytic positive-sign Fourier transform, (visibility, dictionary word)."""
    d = dictionary
    uu, vv = np.asarray(uu)[:, None], np.asarray(vv)[:, None]
    ny, nx = shape
    xc = d.x - (nx * pixel[0] - 1) / 2 + phase_center[0]
    yc = d.y - (ny * pixel[1] - 1) / 2 + phase_center[1]
    determinant = d.a * d.c - d.b**2
    decay = -(np.pi**2) * (d.c * uu**2 - 2 * d.b * uu * vv + d.a * vv**2) / determinant
    return np.asarray(np.exp(decay + 2j * np.pi * (uu * xc + vv * yc)))


def _render(dictionary: _Dictionary, weights: _Array, shape: tuple[int, int], pixel: _Array) -> _Array:
    """Sample surface brightness at pixel centers in NumPy (y, x) order."""
    ny, nx = shape
    xx, yy = np.meshgrid(np.arange(nx) * pixel[0] - 0.5 + pixel[0] / 2, np.arange(ny) * pixel[1] - 0.5 + pixel[1] / 2)
    image = np.zeros(shape)
    d = dictionary
    for i, weight in enumerate(weights):
        x, y = xx - d.x[i], yy - d.y[i]
        image += weight * d.amp[i] * np.exp(-(d.a[i] * x**2 + 2 * d.b[i] * x * y + d.c[i] * y**2))
    return image


def _coordinate_descent(
    dictionary: _Dictionary,
    transform: _Array,
    observed: _Array,
    sigamp: _Array,
    total_flux: float,
    sparseness: float,
    regularization: float,
    maxiter: int,
    abort_numerical_issues: bool,
) -> _Array:
    """SSW 2.0 active-set coordinate descent with a hard continuous flux constraint."""
    d = dictionary
    # SSW intentionally weights by sqrt(SIGAMP), rather than SIGAMP.
    sigma = np.sqrt(sigamp)
    coeff = transform / sigma[:, None] / d.amp
    target = observed / sigma
    basis = np.ascontiguousarray(np.concatenate((coeff.real, coeff.imag), axis=0).T)
    target = np.concatenate((target.real, target.imag))
    ca = -2 * np.sum(basis**2, axis=1)
    atf = total_flux * d.amp / d.viv
    x = np.zeros(len(d.x))
    x[:2] = atf[:2] * 0.5
    residual = x @ basis - target
    xsum = x.sum()
    active = x > 0

    for stage in range(8):
        fraction = stage / 7
        sp = fraction * sparseness + (1 - fraction) * 0.999
        # Avoid an overflowing intermediate for extremely small total fluxes.
        log_lambda = np.log(regularization) + 1 / total_flux + np.log(sp) - np.log1p(-sp) if sp else -np.inf
        if log_lambda > np.log(np.finfo(float).max):
            raise ValueError("Total flux is too small for the SSW VIS-CS regularization scale.")
        penalty = np.exp(log_lambda)
        active_only = False
        for _sweep in range(maxiter):
            moved = False
            indices = np.flatnonzero(active) if active_only else range(len(x))
            for i in indices:
                xi = x[i]
                gap = atf[i] - xi
                if gap <= 0:
                    newx = 0.0
                    epsilon = np.zeros_like(target)
                    bad = True
                else:
                    alpha = 1 / gap
                    epsilon = (residual - basis[i] * xi + target) * alpha
                    divisor = ca[i] + epsilon @ (4 * basis[i] - 2 * epsilon)
                    bad = divisor == 0 or not np.isfinite(divisor)
                    if bad:
                        newx = 0.0
                    else:
                        newx = np.clip(
                            (2 * residual @ (basis[i] - epsilon) - penalty * alpha * (xsum - xi) + penalty) / divisor
                            + xi,
                            0,
                            atf[i] * 0.999,
                        )
                        bad = not np.isfinite(newx)
                        if bad:
                            newx = 0.0
                if bad and abort_numerical_issues and stage == 7:
                    return np.zeros_like(x)
                delta = newx - xi
                if ((newx == 0) != (xi == 0)) or abs(delta) > atf[i] * 0.001:
                    if newx == 0 and xi != 0 and np.count_nonzero(active) < 2:
                        continue
                    if gap <= 0:
                        raise FloatingPointError("VIS-CS reached a singular flux constraint.")
                    active[i] = newx > 0
                    residual += delta * (basis[i] - epsilon)
                    x *= 1 - delta * alpha
                    xsum = (1 - delta * alpha) * xsum - x[i] + newx
                    x[i] = newx
                    moved = True
            if not moved:
                if not active_only:
                    break
                active_only = False
            else:
                active_only = True
        else:
            warnings.warn("VIS-CS coordinate descent reached maxiter before convergence.", RuntimeWarning, stacklevel=2)
    return x / d.amp


def _pair(value: u.Quantity, unit: u.UnitBase, name: str) -> _Array:
    pair = np.asarray(u.Quantity(value).to_value(unit), dtype=float)
    if pair.ndim == 0:
        pair = np.repeat(pair, 2)
    if pair.shape != (2,) or np.any(~np.isfinite(pair)) or np.any(pair <= 0):
        raise ValueError(f"{name} must contain one or two positive finite values.")
    return pair


def vis_cs(
    vis: Visibilities,
    shape: u.Quantity = (65, 65) * u.pix,
    pixel_size: u.Quantity = (1, 1) * u.arcsec / u.pix,
    *,
    total_flux: u.Quantity | None = None,
    sparseness: float = 0.5,
    seed: int | np.random.Generator | None = None,
    n_components: int | None = None,
    iterations: int = 2,
    min_sigma: u.Quantity = 1.5 * u.arcsec,
    max_sigma: u.Quantity = 60 * u.arcsec,
    max_anisotropy: float = 2.5,
    regularization: float = 1000.0,
    maxiter: int = 1000,
    abort_numerical_issues: bool = False,
) -> u.Quantity:
    """Reconstruct visibilities with the SSW VIS-CS 2.0 Gaussian dictionary method.

    Parameters
    ----------
    vis : `~xrayvision.visibility.Visibilities`
        One-dimensional visibilities with finite positive amplitude uncertainties.
        UV coordinates must have inverse angular units.
    shape : `~astropy.units.Quantity`, optional
        Output shape in pixels, in NumPy ``(ny, nx)`` order.
    pixel_size : `~astropy.units.Quantity`, optional
        Scalar or ``(dy, dx)`` angular pixel sizes, matching ``dft_map``.
    total_flux : `~astropy.units.Quantity`, optional
        Positive total flux in visibility units. If omitted, use ``vis.meta['totflux']``
        (a Quantity or numerical array in visibility units), averaging within each
        ``vis.meta['isc']`` detector first, then equally across detectors. Without
        detector labels the arithmetic mean of the flux array is used.
    sparseness : float, optional
        Regularization fraction in [0, 1). Larger values favor fewer Gaussians.
    seed : int or `~numpy.random.Generator`, optional
        Random state for repeatable dictionaries. NumPy and IDL random streams differ.
    n_components : int, optional
        Number of dictionary words; defaults to ``max(2, 200000 // nvis)``.
    iterations : int, optional
        Dictionary generations, including the initial dictionary (default two).
    min_sigma, max_sigma : `~astropy.units.Quantity`, optional
        Gaussian sigma limits. Defaults 1.5 and 60 arcsec follow installed SSW 2.0.
    max_anisotropy : float, optional
        Axis-ratio bound (default 2.5); one gives circular Gaussians.
    regularization : float, optional
        SSW penalty scale (default 1000), applied to the numerical flux values in
        the input visibility unit. Use consistent flux units across comparisons.
    maxiter : int, optional
        Maximum full/active-set sweeps per each of eight regularization stages.
        Emits a RuntimeWarning if a stage does not converge.
    abort_numerical_issues : bool, optional
        Return zero coefficients for a singular final-stage update, as in SSW.

    Returns
    -------
    image : `~astropy.units.Quantity`
        Nonnegative surface brightness in visibility units per square arcsecond.
        Multiply by pixel area before using ``dft_map``. Output is sampled about
        ``vis.phase_center`` (in ``(y, x)`` order) using the positive Fourier sign
        of ``dft_map``.

    Notes
    -----
    This follows the implementation of VIS-CS 2.0 shipped in SolarSoft, including
    sqrt(SIGAMP) weighting, eight sparsity stages, continuous in-view total-flux
    constraint, active-set descent, dictionary refinement and up-to-200-word pruning.
    The paper describes sigma up to 40 arcsec; SSW uses 60 arcsec. NumPy uses a
    different random generator and float64 rendering (SSW renders float32).
    Flux is constrained before pruning and continuous integration differs from
    summing pixel samples, so the returned pixel sum need not equal total_flux.

    References
    ----------
    Felix, Bolzern & Battaglia (2017), ApJ 849, 10,
    https://doi.org/10.3847/1538-4357/aa8d1e.
    """
    dims = _pair(shape, u.pix, "shape")
    if np.any(dims != np.floor(dims)):
        raise ValueError("shape must contain integer pixel counts.")
    image_shape = (int(dims[0]), int(dims[1]))
    pixel = _pair(pixel_size, u.arcsec / u.pix, "pixel_size")[::-1]
    lo = u.Quantity(min_sigma).to_value(u.arcsec)
    hi = u.Quantity(max_sigma).to_value(u.arcsec)
    if np.ndim(lo) or np.ndim(hi) or not (0 < lo <= hi < np.inf):
        raise ValueError("Require finite 0 < min_sigma <= max_sigma.")
    if not np.isfinite(sparseness) or not 0 <= sparseness < 1:
        raise ValueError("sparseness must be in [0, 1).")
    if not np.isfinite(max_anisotropy) or max_anisotropy < 1:
        raise ValueError("max_anisotropy must be finite and at least one.")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive.")
    for name, value, minimum in (("iterations", iterations, 1), ("maxiter", maxiter, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be a positive integer.")
    unit = vis.visibilities.unit
    observed = np.asarray(vis.visibilities.to_value(unit), dtype=complex)
    uu, vv = vis.u.to_value(1 / u.arcsec), vis.v.to_value(1 / u.arcsec)
    uncertainty = vis.amplitude_uncertainty
    if uncertainty is None:
        raise ValueError("VIS-CS requires amplitude uncertainties.")
    sigamp = np.asarray(uncertainty.to_value(unit))
    if observed.ndim != 1 or len(observed) < 2 or uu.shape != observed.shape or vv.shape != observed.shape:
        raise ValueError("VIS-CS requires at least two one-dimensional visibility samples.")
    if (
        sigamp.shape != observed.shape
        or np.iscomplexobj(sigamp)
        or np.any(sigamp <= 0)
        or any(np.any(~np.isfinite(t)) for t in (observed, uu, vv, sigamp))
    ):
        raise ValueError(
            "Visibilities and UV coordinates must be finite, with positive finite amplitude uncertainties."
        )
    if total_flux is None:
        meta = vis.meta
        if not isinstance(meta, Mapping) or "totflux" not in meta:
            raise ValueError("Provide total_flux or vis.meta['totflux']; VIS-CS requires a total-flux constraint.")
        fluxes = meta["totflux"]
        fluxes = np.asarray(fluxes.to_value(unit) if isinstance(fluxes, u.Quantity) else fluxes)
        if fluxes.shape != observed.shape or np.any(~np.isfinite(fluxes)) or np.any(fluxes <= 0):
            raise ValueError("meta['totflux'] must contain one positive finite flux per visibility.")
        if "isc" in meta:
            labels = np.asarray(meta["isc"])
            if labels.shape != observed.shape:
                raise ValueError("meta['isc'] must have one detector label per visibility.")
            flux = np.mean([fluxes[labels == label].mean() for label in np.unique(labels)])
        else:
            flux = fluxes.mean()
    else:
        flux = u.Quantity(total_flux).to_value(unit)
    if np.ndim(flux) or not np.isfinite(flux) or flux <= 0:
        raise ValueError("total_flux must be a positive finite scalar.")
    if n_components is None:
        n_components = max(2, 200000 // len(observed))
    if isinstance(n_components, bool) or not isinstance(n_components, Integral) or n_components < 2:
        raise ValueError("n_components must be an integer of at least two.")
    center = np.asarray(vis.phase_center.to_value(u.arcsec))[::-1]
    if center.shape != (2,) or np.any(~np.isfinite(center)):
        raise ValueError("vis.phase_center must contain two finite angular coordinates.")
    rng = np.random.default_rng(seed)
    dictionary = _random_dictionary(rng, n_components, image_shape, pixel, lo, hi, max_anisotropy)
    for generation in range(iterations):
        transform = _fourier(dictionary, uu, vv, image_shape, pixel, center)
        weights = _coordinate_descent(
            dictionary,
            transform,
            observed,
            sigamp,
            float(flux),
            sparseness,
            regularization,
            maxiter,
            abort_numerical_issues,
        )
        limit = max(1e-8, np.partition(weights, -201)[-201]) if len(weights) > 200 else 1e-8
        selected = np.flatnonzero(weights > limit)
        if not len(selected):
            return np.zeros(image_shape) * unit / u.arcsec**2
        if generation == iterations - 1:
            return _render(dictionary.take(selected), weights[selected], image_shape, pixel) * unit / u.arcsec**2
        dictionary = _refine_dictionary(
            dictionary.take(selected), rng, n_components, image_shape, pixel, lo, hi, max_anisotropy
        )
