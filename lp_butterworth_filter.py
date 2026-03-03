# ============================================================
# lp_filter.py
# Standalone 2nd-order Butterworth Low-Pass Filter (JAX)
# Input  : noise (T,)
# Output : filtered noise (T,)
# ============================================================

import jax
import jax.numpy as jnp
from functools import partial


# ------------------------------------------------------------
# Compute 2nd-order Butterworth coefficients
# ------------------------------------------------------------
def butterworth_coefficients(dt, fc):
    """
    dt : sampling time (seconds)
    fc : cutoff frequency (Hz)
    """

    alpha = jnp.tan(jnp.pi * fc * dt)

    denom = 1.0 + jnp.sqrt(2.0) * alpha + alpha**2

    b0 = alpha**2 / denom
    b1 = 2.0 * b0
    b2 = b0

    a1 = 2.0 * (alpha**2 - 1.0) / denom
    a2 = (1.0 - jnp.sqrt(2.0) * alpha + alpha**2) / denom

    return b0, b1, b2, a1, a2


# ------------------------------------------------------------
# Standalone filter function
# ------------------------------------------------------------
@partial(jax.jit, static_argnums=(1, 2))
def lowpass_filter(noise, dt, fc):
    """
    noise : (T,)
    dt    : sampling time
    fc    : cutoff frequency

    returns filtered_noise : (T,)
    """

    b0, b1, b2, a1, a2 = butterworth_coefficients(dt, fc)

    def step(carry, x):
        x1, x2, y1, y2 = carry

        y = (
            b0 * x +
            b1 * x1 +
            b2 * x2 -
            a1 * y1 -
            a2 * y2
        )

        new_carry = (x, x1, y, y1)
        return new_carry, y

    # initial conditions
    init = (0.0, 0.0, 0.0, 0.0)

    _, filtered = jax.lax.scan(step, init, noise)

    return filtered