"""
blasius_gaussian.py
===================

Semi-analytic approximation to the Blasius laminar boundary layer, and to the
thermal boundary layer with radiation, from

    Moeini, M. and M. R. Chamani (2017). New perspectives on the laminar
    boundary layer physics in a polarized pressure field with temperature
    gradient: an analytical approximation to Blasius equation.
    Journal of Applied Fluid Mechanics, 10(4), 1071-1077.
    https://doi.org/10.18869/acadpub.jafm.73.241.27544

Equation numbers in the comments refer to that paper.
Needs NumPy and SciPy (SciPy only for the error function and its inverse).

Notation: eta = y sqrt(U0 / (nu x)) is the Blasius similarity variable and
f'(eta) = u / U0. The paper approximates f''(eta) by an amended Gaussian
probability density function with zero mean and standard deviation
sigma = 2.18 (Eqs. 9-10), which gives the closed form of Eq. 11.
"""

import numpy as np
from scipy.special import erf, erfinv

SIGMA = 2.18          # standard deviation of the amended Gaussian, Eq. 10


def f_double_prime(eta, sigma=SIGMA):
    """f''(eta): the amended Gaussian PDF of Eqs. 9-10, for eta >= 0."""
    return np.sqrt(2 / np.pi) / sigma * np.exp(-eta**2 / (2 * sigma**2))


def f_prime(eta, sigma=SIGMA):
    """f'(eta) = u / U0, the integral of Eq. 10."""
    return erf(eta / (sigma * np.sqrt(2)))


def f(eta, sigma=SIGMA):
    """f(eta), the closed-form approximation of Eq. 11."""
    c = sigma * np.sqrt(2 / np.pi)
    return -c + c * np.exp(-(eta / (sigma * np.sqrt(2)))**2) + eta * erf(eta / (sigma * np.sqrt(2)))


def velocity(x, y, U0, nu, sigma=SIGMA):
    """
    Velocity components at (x, y) as printed in Eq. 12.
    u follows from Eq. 11; v is the expression obtained from the momentum
    equation (Eq. 1) and plotted in Fig. 3.
    """
    eta = y * np.sqrt(U0 / (nu * x))
    u = U0 * erf(eta / (np.sqrt(2) * sigma))
    v = 0.5 * U0 * y / x * (erf(eta / (np.sqrt(2) * sigma)) - 2 / sigma**2)
    return u, v


def velocity_continuity(x, y, U0, nu, sigma=SIGMA):
    """Same u, with v from the continuity-based relation of Eq. 13, v = 0.5 sqrt(U0 nu / x) (eta f' - f)."""
    eta = y * np.sqrt(U0 / (nu * x))
    u = U0 * f_prime(eta, sigma)
    v = 0.5 * np.sqrt(U0 * nu / x) * (eta * f_prime(eta, sigma) - f(eta, sigma))
    return u, v


def boundary_layer_coefficient(u_ratio=erf(3.0), sigma=SIGMA):
    """
    Coefficient a in the boundary-layer locus y = a sqrt(nu x / U0), Eqs. 15-18.
    u_ratio is the value of u / U0 that defines the edge. The paper uses
    c0 = erf(3) = 0.999980, which gives a = 9.24895 (Eq. 18); the usual
    0.99 criterion gives about 5.6, against Schlichting's 5.0.
    """
    return sigma * np.sqrt(2) * erfinv(u_ratio)


def diffusion_constants(U0, nu, alpha, k0=1.0, sigma=SIGMA):
    """Uniform-pressure momentum constant D_b (Eq. 19), thermal constant D_t (Eq. 22) and modified Prandtl number (Eq. 23)."""
    D_b = sigma**2 * nu / (2 * U0)
    D_t = alpha * sigma**2 / (2 * k0 * U0)
    Pr_bar = D_b / D_t
    return D_b, D_t, Pr_bar


def theta_prime_0(Pr, k0=1.0, sigma=SIGMA):
    """
    Wall temperature gradient theta'(0) from the energy equation, Eq. 5,
        theta'' + (Pr k0 / 2) f theta' = 0,  theta(0) = 0, theta(inf) = 1 (Eq. 7),
    with f from Eq. 11. The equation is linear in theta, so
        theta'(eta) = theta'(0) exp(-(Pr k0 / 2) F(eta)),   F(eta) = integral of f from 0 to eta,
    and theta(inf) = 1 gives theta'(0) = 1 / integral of exp(-(Pr k0 / 2) F) from 0 to infinity.
    Both integrals use the trapezoidal rule on a fine grid.
    """
    eta = np.arange(0.0, 60.0, 0.001)
    d = eta[1] - eta[0]
    fv = f(eta, sigma)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (fv[1:] + fv[:-1]) * d)])
    g = np.exp(-0.5 * Pr * k0 * F)
    return 1.0 / np.sum(0.5 * (g[1:] + g[:-1]) * d)


if __name__ == "__main__":
    print("Amended Gaussian with sigma = %.2f: f''(0) = %.5f  (Blasius: 0.33206)" % (SIGMA, f_double_prime(0.0)))

    # Fig. 2: closed form of Eq. 11 against Howarth's (1938) numerical solution of the Blasius equation.
    howarth = {0: (0.00000, 0.00000, 0.33206), 1: (0.16557, 0.32979, 0.32301),
               2: (0.65003, 0.62977, 0.26675), 3: (1.39682, 0.84605, 0.16136),
               4: (2.30576, 0.95552, 0.06424), 5: (3.28329, 0.99155, 0.01591),
               6: (4.27964, 0.99898, 0.00240), 8: (6.27923, 1.00000, 0.00001)}
    print("\n eta   f Eq.11  f Howarth   f'=u/U0  f' Howarth   f''     f'' Howarth")
    for eta, (fh, fph, fpph) in howarth.items():
        print(" %3d   %7.5f  %7.5f     %7.5f  %7.5f     %7.5f  %7.5f"
              % (eta, f(eta), fh, f_prime(eta), fph, f_double_prime(eta), fpph))

    # Eq. 18: boundary-layer locus y = a sqrt(nu x / U0).
    print("\nBoundary-layer edge, coefficient a in y = a sqrt(nu x / U0):")
    print("  u = 0.999980 U0 (this paper, Eq. 18): a = %.5f" % boundary_layer_coefficient(erf(3.0)))
    print("  u = 0.99 U0 (usual criterion)        : a = %.2f   (Schlichting 1979: 5.0)" % boundary_layer_coefficient(0.99))

    # Fig. 3: velocity field for U0 = 0.1 m/s and nu = 1e-6 m2/s, evaluated at one point.
    U0, nu, x, y = 0.1, 1.0e-6, 4.0, 0.01
    u, v = velocity(x, y, U0, nu)
    u2, v2 = velocity_continuity(x, y, U0, nu)
    print("\nVelocity at x = %.1f m, y = %.3f m for U0 = %.1f m/s, nu = %.0e m2/s:" % (x, y, U0, nu))
    print("  u/U0 = %.4f,  v/U0 = %.2e   (Eq. 12, v from the momentum equation)" % (u / U0, v / U0))
    print("  u/U0 = %.4f,  v/U0 = %.2e   (Eq. 13, v from continuity)" % (u2 / U0, v2 / U0))
    print("  boundary-layer edge at this x: y = %.4f m   (Eq. 18)" % (boundary_layer_coefficient() * np.sqrt(nu * x / U0)))

    # Eqs. 19, 22 and 23 for the same flow, Pr = 0.7 and k0 = 1.
    D_b, D_t, Pr_bar = diffusion_constants(U0, nu, alpha=nu / 0.7, k0=1.0)
    print("\nD_b = %.3e m,  D_t = %.3e m,  modified Prandtl number = %.2f" % (D_b, D_t, Pr_bar))

    # Table 1: theta'(0) for k0 = 1, compared with the paper and with Bataller (2008).
    paper = {0.1: 0.14035, 0.4: 0.24094, 0.7: 0.29607, 5: 0.59023, 10: 0.74735, 50: 1.28518, 100: 1.62100}
    bataller = {0.7: 0.29268, 5: 0.57669, 10: 0.72814, 50: 1.24729, 100: 1.57183}
    print("\nTable 1: wall temperature gradient theta'(0), k0 = 1")
    print("   Pr    this code   paper     Bataller (2008)")
    for Pr in paper:
        ref = "%.5f" % bataller[Pr] if Pr in bataller else "-"
        print("  %5.1f   %.5f    %.5f   %s" % (Pr, theta_prime_0(Pr), paper[Pr], ref))
