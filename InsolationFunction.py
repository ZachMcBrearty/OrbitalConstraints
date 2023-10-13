# q_0 = 1360 Wm^-2 at a = 1 AU
# Z = solar zenith angle (?)
# S = q_0 cos(Z) = q_0 μ
# latitude - θ, solar declination - δ
# obliquity - δ_0, Orbital Longitude - L_s
# μ = sin(θ)sin(δ) + cos(θ)cos(δ)cos(h)
# sin(δ) = -sin(δ_0)cos(L_s + π/2)
# L_s = ωt, ω prop a^-3/2
# diurnally averaged: S = q_0 avg(μ)
# S = q_0 / π * (1AU / a)^2 * (H sinθ sinδ + cosθ, cosδ sinH)

import numpy as np
import numpy.typing as npt

floatarr = npt.NDArray[np.float64]


def delta(a: float, t: float | floatarr, delta_0: float) -> float | floatarr:
    """Combining equations A2, A3, and partially A4 to get δ
    a: semi-major axis, AU
    t: time, years
    delta_0: obliquity, radians

    returns: δ, the solar declination, -π/2 <= δ <= π/2
    """
    omega = 2 * np.pi * a ** (-3 / 2)
    L_s = omega * t
    return np.arcsin(-np.sin(delta_0) * np.cos(L_s + np.pi / 2))


def H(theta: float | floatarr, delta_: float | floatarr) -> float | floatarr:
    """implementing eqn A5 to find H
    theta: planetary latitude, radians
    delta_: solar declination, radians

    returns: 0 < H < π, radian half-day length"""
    q = -np.tan(theta) * np.tan(delta_)
    if isinstance(q, np.ndarray):
        q[q > 1] = 1
        q[q < -1] = -1
    else:
        if q > 1:
            q = 1
        elif q < -1:
            q = -1
    return np.arccos(q)


def S(
    a: float, theta: float | floatarr, t: float | floatarr, delta_0: float
) -> float | floatarr:
    """implements equation A8 from appendix A of WK97
    a: semi-major axis, AU
    theta: planetary latitude, radians
    t: float: years
    delta_0: obliquity, radians

    returns: S, solar insolation, J s^-1 m^-2"""
    q_0 = 1360  # Wm^-2
    delta_ = delta(a, t, delta_0)
    cosdelta = np.cos(delta_)  #
    sindelta = np.sin(delta_)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    H_ = H(theta, delta_)
    return (
        q_0
        / np.pi
        * a**-2
        * (H_ * sintheta * sindelta + costheta * cosdelta * np.sin(H_))
    )


def dist_adv(
    a: float, e: float, t: float | floatarr, offset: float = 0, iter: int = 3
) -> float | floatarr:
    """a: semimajor axis, AU
    e: eccentricity, 0 < e < 1
    t: time, years
    offset: temporal offset from vernal equinox, years
    iter: number of iterations of newton's method
        - 3 iterations gives an error of ~5% at e = 0.9

    returns: distance to star, AU"""
    T = a ** (3 / 2)
    M = 2 * np.pi * ((t + offset) % T)
    E = M
    for _ in range(iter):
        E = E + ((M + e * np.sin(E) - E) / (1 - e * np.cos(E)))
    return a * (1 - e * np.cos(E))


def dist_basic(
    a: float, e: float, t: float | floatarr, offset: float = 0
) -> float | floatarr:
    """a: semimajor axis, AU
    e: eccentricity, 0 < e < 1
    t: time, years
    offset: temporal offset from vernal equinox, years

    returns: distance to star, AU"""
    T = a ** (3 / 2)
    M = 2 * np.pi * ((t + offset) % T)
    E_0 = M
    E_1 = M + e * np.sin(E_0)
    E_2 = M + e * np.sin(E_1)

    return a * (1 - e * np.cos(E_2))


def dist(
    a: float, e: float, t: float | floatarr, offset: float = 0, iter_num: int = 3
) -> float | floatarr:
    """Chooses dist_basic (e <= 0.5) or dist_adv (e > 0.5)

    a: semimajor axis, AU
    e: eccentricity, 0 < e < 1
    t: time, years
    offset: temporal offset from vernal equinox, years
    iter_num: number of iterations of newton's method if e > 0.5

    returns: distance to star, AU"""
    if e > 0.5:
        return dist_adv(a, e, t, offset, iter_num)
    else:
        return dist_basic(a, e, t, offset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    time = np.linspace(2, 4, 300)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    e = 0.9
    r = dist(1, e, time)
    r_prime_0 = dist_adv(1, e, time, 0)
    r_prime_1 = dist_adv(1, e, time, 1)
    r_prime_2 = dist_adv(1, e, time, 2)
    r_prime_3 = dist_adv(1, e, time, 3)

    r_prime_100 = dist_adv(1, e, time, 100)

    for lat in range(0, 91, 15):
        eq = S(r, np.deg2rad(lat), time, np.deg2rad(0))
        a = ax1.plot(time, eq, label=f"{lat} deg")

    ax2.plot(time, r, label=f"basic")
    ax2.plot(time, r_prime_0, label=f"adv, 0")
    ax2.plot(time, r_prime_1, label=f"adv, 1")
    ax2.plot(time, r_prime_2, label=f"adv, 2")
    ax2.plot(time, r_prime_3, label=f"adv, 3")
    ax2.plot(time, r_prime_100, label=f"adv, 100")

    # ax3.plot(time, np.abs(r_prime_100 - r), label="adv_100 - basic")
    # ax3.plot(time, np.abs(r_prime_100 - r_prime_0), label="adv_100 - adv_0")
    # ax3.plot(time, np.abs(r_prime_100 - r_prime_1), label="adv_100 - adv_1")
    # ax3.plot(time, np.abs(r_prime_100 - r_prime_2), label="adv_100 - adv_2")
    ax3.plot(time, np.abs(r_prime_100 - r_prime_3), label="adv_100 - adv_3")
    # plt.axhline(np.average(eq), color=a[0].get_c(), label=f"{lat} deg avg")  # type: ignore

    ax2.set_xlabel("time, yrs")
    ax1.set_ylabel("Insolation, W m$^{-2}$")

    ax2.set_ylabel("distance from sun, r, AU")
    ax2.legend()
    ax3.legend()
    plt.show()
