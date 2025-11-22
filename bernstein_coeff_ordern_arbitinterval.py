# Author: Alinjar Dan
# Email: alinjardannitdgp2014@gmail.com
# GitHub:https://github.com/alinjar1996
import numpy as np
from scipy.special import binom

def bernstein_coeff_ordern_new(n, tmin, tmax, t_actual):
    l = tmax - tmin
    t = (t_actual - tmin) / l  # Normalizing the time to [0, 1]

    # Bernstein polynomial coefficients
    P = np.array([binom(n, i) * (1 - t)**(n - i) * t**i for i in range(n + 1)]).squeeze().T

    # First derivative of the Bernstein polynomial (Pdot)
    Pdot = np.array([
        n * (binom(n - 1, i - 1) * (1 - t)**(n - i) * t**(i - 1) if i > 0 else 0) -
        n * (binom(n - 1, i) * (1 - t)**(n - i - 1) * t**i if i < n else 0)
        for i in range(n + 1)
    ]).squeeze().T / l

    # Second derivative of the Bernstein polynomial (Pddot)
    Pddot = np.array([
        n * (n - 1) * (
            (binom(n - 2, i - 2) * (1 - t)**(n - i) * t**(i - 2) if i > 1 else 0) -
            2 * (binom(n - 2, i - 1) * (1 - t)**(n - i - 1) * t**(i - 1) if 0 < i < n else 0) +
            (binom(n - 2, i) * (1 - t)**(n - i - 2) * t**i if i < n - 1 else 0)
        )
        for i in range(n + 1)
    ]).squeeze().T / (l**2)

    #Integral of the Bernstein polynomial (Pint)
    n_plus_1 = n + 1
    P_n_plus_1 = np.array([
        binom(n_plus_1, j) * (1 - t)**(n_plus_1 - j) * t**j
        for j in range(n_plus_1 + 1)
    ])
    
    # The normalized integral (Pint_norm) of B_{i,n}(t) from 0 to t is:
    # integral_0^t B_{i,n}(\tau) d\tau = 1/(n+1) * Sum_{j=i+1}^{n+1} B_{j, n+1}(t)
    Pint_norm = np.zeros(n + 1)
    for i in range(n + 1):
        # Sum B_{j, n+1}(t) for j from i+1 up to n+1.
        # Python slicing P_n_plus_1[i + 1:] corresponds to j >= i + 1.
        Pint_norm[i] = (1 / n_plus_1) * np.sum(P_n_plus_1[i + 1:])

    # Denormalize: Pint = l * Pint_norm
    # Since d(t_actual) = l * d(t)
    Pint = l * Pint_norm

    return P, Pdot, Pddot, Pint
