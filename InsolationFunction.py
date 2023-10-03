# q_0 = 1360 Wm^-2 at a = 1 AU
# Z = solar zenith angle (?)
# S = q_0 cos(Z) = q_0 μ
# latitude - θ, solar declination - δ
# obliquity - δ_0, Orbital Longitude - L_s
# μ = sin(θ)sin(δ) + cos(θ)cos(δ)cos(h)
# sin(δ) = -sin(δ_0)cos(L_s + π/2)
# L_s = ωt, ω prop a^-3/2
# diurnally averaged: S = q_0 μ^bar
# S = q_0 / π * (1AU / a)^2 * (H sinθ sinδ + cosθ,cosδ sinH)

import numpy as np
import numpy.typing as npt
floatarr = npt.NDArray[np.float64]

def delta(a: float, t: float|floatarr, delta_0: float) -> float|floatarr:
    '''Combining equations A2, A3, and partially A4 to get δ
    a: float: semi-major axis, AU
    t: float: time, years 
    delta_0: float: obliquity, radians

    returns: float: δ, the solar declination, -π/2 <= δ <= π/2
    '''
    omega = 2*np.pi * a**(-3/2)
    L_s = omega * t
    return np.arcsin(
        -np.sin(delta_0)*np.cos(L_s + np.pi/2)
    )

def H(theta:float , delta_:float|floatarr) -> float|floatarr:
    '''implementing eqn A5 to find H
    theta: planetary latitude, radians    
    delta_: solar declination, radians
    
    returns: 0 < H < π, radian half-day length'''
    q = -np.tan(theta) * np.tan(delta_)
    if isinstance(q, np.ndarray):
        q[q>1] = 1
        q[q<-1] = -1
    else:
        if q > 1:
            q = 1
        elif q < -1:
            q = -1
    return np.arccos(q)

def S(a:float, theta:float, t:float|floatarr, delta_0:float) -> float|floatarr:
    '''implements equation A8 from appendix A of WK97
    a: float: semi-major axis, AU
    theta: float: planetary latitude, radians
    t: float: time, years
    delta_0: float: obliquity, radians
    
    returns: float: S, solar insolation, W m^-2'''
    q_0 = 1360 # Wm^-2
    delta_ = delta(a, t, delta_0)
    cosdelta = np.cos(delta_)#
    sindelta = np.sin(delta_)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    H_ = H(theta, delta_)
    return q_0 / np.pi * a**-2 * (H_*sintheta*sindelta + costheta*cosdelta*np.sin(H_))

import matplotlib.pyplot as plt

time = np.linspace(0, 2, 100)

for lat in range(0, 91, 15):
    eq = S(1, np.deg2rad(lat), time, np.deg2rad(90))
    a = plt.plot(time, eq, label=f"{lat} deg")
    plt.axhline(np.average(eq), color=a[0].get_c(), label=f"{lat} deg avg") # type: ignore

plt.xlabel("time, yrs")
plt.ylabel("Insolation, W m$^{-2}$")
plt.legend()
plt.show()