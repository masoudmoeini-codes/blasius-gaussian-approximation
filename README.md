# blasius-gaussian-approximation

Closed-form approximation to the Blasius laminar boundary layer, together with the thermal boundary layer under radiation.

This is the reference implementation of the equations in

> Moeini, M. and Chamani, M. R. (2017). New Perspectives on the Laminar Boundary Layer Physics in a Polarized Pressure Field with Temperature Gradient: an Analytical Approximation to Blasius Equation. *Journal of Applied Fluid Mechanics*, 10(4), 1071–1077. https://doi.org/10.18869/acadpub.jafm.73.241.27544

The paper treats f''(η) of the Blasius equation as a probability density function. It approximates it by a Gaussian with zero mean and standard deviation σ = 2.18, cut at η = 0 and doubled. The result is a closed-form velocity field, a new estimate of the boundary-layer edge and diffusion-type forms of the momentum and energy equations. The code is a single file that depends on NumPy and SciPy.

## What is implemented

| Function | Paper | Purpose |
|---|---|---|
| `f_double_prime` | Eqs. 9–10 | amended Gaussian approximation of f''(η) |
| `f_prime` | Eq. 11, derivative | f'(η) = u/U₀ |
| `f` | Eq. 11 | closed-form f(η) |
| `velocity` | Eq. 12 | u and v at a point, with v from the momentum equation (Fig. 3) |
| `velocity_continuity` | Eq. 13 | u and v at a point, with v from continuity |
| `boundary_layer_coefficient` | Eqs. 15–18 | coefficient a in the boundary-layer locus y = a √(νx/U₀) |
| `diffusion_constants` | Eqs. 19, 22, 23 | D_b, D_t and the modified Prandtl number |
| `theta_prime_0` | Eqs. 5, 7 | wall temperature gradient θ'(0), Table 1 |

## Quick start

```bash
pip install numpy scipy
python blasius_gaussian.py
```

Running the file reproduces the numbers of the paper. It prints f''(0), the comparison with Howarth's numerical solution (Fig. 2), the boundary-layer coefficient 9.24895 of Eq. 18, the velocity field of Fig. 3 at one point, the diffusion constants and Table 1.

```
Amended Gaussian with sigma = 2.18: f''(0) = 0.36600  (Blasius: 0.33206)

 eta   f Eq.11  f Howarth   f'=u/U0  f' Howarth   f''     f'' Howarth
   0   0.00000  0.00000     0.00000  0.00000     0.36600  0.33206
   1   0.17986  0.16557     0.35356  0.32979     0.32945  0.32301
   2   0.68468  0.65003     0.64108  0.62977     0.24028  0.26675
   3   1.42907  1.39682     0.83122  0.84605     0.14199  0.16136
   4   2.31760  2.30576     0.93347  0.95552     0.06798  0.06424
   5   3.27688  3.28329     0.97819  0.99155     0.02637  0.01591
   6   4.26450  4.27964     0.99408  0.99898     0.00829  0.00240
   8   6.26074  6.27923     0.99976  1.00000     0.00044  0.00001

Boundary-layer edge, coefficient a in y = a sqrt(nu x / U0):
  u = 0.999980 U0 (this paper, Eq. 18): a = 9.24896
  u = 0.99 U0 (usual criterion)        : a = 5.62   (Schlichting 1979: 5.0)

Velocity at x = 4.0 m, y = 0.010 m for U0 = 0.1 m/s, nu = 1e-06 m2/s:
  u/U0 = 0.5317,  v/U0 = 1.39e-04   (Eq. 12, v from the momentum equation)
  u/U0 = 0.5317,  v/U0 = 3.18e-04   (Eq. 13, v from continuity)
  boundary-layer edge at this x: y = 0.0585 m   (Eq. 18)

D_b = 2.376e-05 m,  D_t = 3.395e-05 m,  modified Prandtl number = 0.70

Table 1: wall temperature gradient theta'(0), k0 = 1
   Pr    this code   paper     Bataller (2008)
    0.1   0.14036    0.14035   -
    0.4   0.24094    0.24094   -
    0.7   0.29608    0.29607   0.29268
    5.0   0.59023    0.59023   0.57669
   10.0   0.74735    0.74735   0.72814
   50.0   1.28518    1.28518   1.24729
  100.0   1.62100    1.62100   1.57183
```

The functions can also be imported and used with your own values, as in the example below.

```python
from blasius_gaussian import velocity, theta_prime_0

u, v = velocity(x=2.0, y=0.02, U0=0.1, nu=1e-6)   # m/s
wall_gradient = theta_prime_0(Pr=0.7, k0=1.0)
```

## A note on the crosswise velocity

The paper gives two expressions for v. Eq. 12 is derived from the momentum equation and is the one plotted in Fig. 3. Eq. 13 is the continuity-based relation used in the Blasius literature. With the exact Blasius profile the two coincide. With the approximate profile of Eq. 11 they differ, because that profile does not satisfy the momentum and the continuity equations at the same time. The script prints both so that readers can compare them.

## Citing

Please cite the paper above. The code itself can be referred to by its repository address, https://github.com/USERNAME/blasius-gaussian-approximation. GitHub's "Cite this repository" button gives the reference in BibTeX and APA format.

## License

MIT, see [LICENSE](LICENSE).
