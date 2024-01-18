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

from Constants import *


def delta(a: float | floatarr, t: float | floatarr, delta_0: float) -> float | floatarr:
    """Combining equations A2, A3, and partially A4 to get δ
    a: semi-major axis, AU
    t: time, years
    delta_0: obliquity, radians

    returns: δ, the solar declination, -π/2 <= δ <= π/2
    """
    # T^2 = a^3, ω = 2π/T -> ω = 2π / a^(3/2)
    omega = 2 * np.pi * a ** (-3 / 2)
    L_s = (omega * t + np.pi / 2) % (2 * np.pi)
    # sinδ = -sin(δ_0) cos(L_s + π/2)
    return np.arcsin(-np.sin(delta_0) * np.cos(L_s))


def H(theta: float | floatarr, delta_: float | floatarr) -> float | floatarr:
    """implementing eqn A5 to find H
    theta: planetary latitude, radians
    delta_: solar declination, radians

    returns: 0 < H < π, radian half-day length"""
    # cos H = -tanθ tanδ,
    # however tan is unbounded, so approximate by saying if the product
    # is > 1 then set it to 1 (i.e. H = 0), and < -1 set it to -1 (i.e. H = pi)
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
    a: float,
    theta: float | floatarr,
    t: float | floatarr,
    delta_0: float,
    e: float,
    A_gas: float,
    rad_gas: float,
    r_moon_to_gas: float,
    offset: float = 0,
) -> float | floatarr:
    """implements equation A8 from appendix A of WK97
    a: semi-major axis, AU
    theta: planetary latitude, radians
    t: float: years
    delta_0: obliquity, radians
    e: eccentricity
    A_gas: albedo of the gas giant
    rad_gas: radius of gas giant, R_J
    r_moon_to_gas: distance from gas giant to the moon, AU

    returns: S, solar insolation, J s^-1 m^-2"""
    q_0 = 1360 * (
        1
        + (1 - A_gas)
        * (rad_gas * RADIUS["jupiter"]) ** 2
        / (4 * (AU * r_moon_to_gas) ** 2)
    )  # Wm^-2
    delta_ = delta(a, t, delta_0)
    cosdelta = np.cos(delta_)
    sindelta = np.sin(delta_)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    H_ = H(theta, delta_)
    r = dist(a, e, t, offset, 5)
    # S = q_0 / π * a^-2 * (H sinθ sinδ + cosθ cosδ sinH)
    return (
        q_0
        / np.pi
        * r**-2
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
    # modulo as M is periodic in T
    M = 2 * np.pi * ((t + offset) % T)
    # M = E - e SinE
    E = M
    for _ in range(iter):
        # E_i+1 = E_i + (M + e sin(E_i) - E_i) / (1 - e cos(E_i))
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
    # E_i+1 = M + e sin(E_i)
    E_1 = M + e * np.sin(E_0)
    E_2 = M + e * np.sin(E_1)

    return a * (1 - e * np.cos(E_2))


def dist(
    a: float, e: float, t: float | floatarr, offset: float = 0, iter_num: int = 4
) -> float | floatarr:
    """Chooses dist_basic (e <= 0.5) or dist_adv (e > 0.5)

    a: semimajor axis, AU
    e: eccentricity, 0 < e < 1
    t: time, years
    offset: temporal offset from vernal equinox, years
    iter_num: number of iterations of newton's method if e > 0.5

    returns: distance to star, AU"""
    if e >= 0:
        return dist_adv(a, e, t, offset, iter_num)
    else:
        return dist_basic(a, e, t, offset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    time = np.linspace(0, 1, 10)
    a = 1
    e = 0  # 0.0167
    delta_0 = np.deg2rad(23.5)
    delta_0s = np.deg2rad(np.linspace(0, 90, 361))
    off = 0.0
    for t in time:
        insols = []
        for delta_0 in delta_0s:
            insols.append(S(a, -np.pi / 2, t, delta_0, e, 0, 0, 0.01))
        plt.plot(np.rad2deg(delta_0s), insols, label=f"t={round(t, 2)}")
    # plt.plot(time, insols)

    # r = dist(1, e, time, offset=off)
    # for lat in range(0, 91, 15):
    #     eq = S(r, np.deg2rad(lat), time, np.deg2rad(delta_0))
    #     a = plt.plot(time, eq, label=f"{lat} deg")
    #     plt.axhline(np.average(eq), color=a[0].get_c(), label=f"{lat} deg avg")  # type: ignore
    # plt.title(rf"offset = {off}yr, eccentricity = {e}, $\delta$ = {delta_0}")
    plt.xlabel("delta_0, degs")
    plt.ylabel("Insolation, W m$^{-2}$")

    plt.legend()
    plt.show()
