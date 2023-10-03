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

from numpy import pi as np_pi, sin as np_sin, cos as np_cos
import numpy as np
def delta(a: float, t: float, delta_0: float):
    '''a: float: semi-major axis, AU
    t: float: time, seconds 
    delta_0: float: obliquity, radians
    '''
    omega = 2*np.pi * a**(-3/2)
    L_s = omega * t
    return np.arcsin(
        -np.sin(delta_0)*np.cos(L_s + np.pi/2)
    )

def S(a:float, H:float, theta:float, t:float, delta_0:float):
    '''implements equation A8 from appendix A of WK97
    a: float: semi-major axis, AU
    H: float: radian half-day length, radians, 0 < H < π
    theta: float: planetary latitude, radians
        - also called lambda
    t: float: time, seconds(?)
    delta_0: float: obliquity, radians'''
    q_0 = 1360 # Wm^-2
    cosdelta = np.cos(delta(a, t, delta_0))
    return q_0 / np.pi * a**-2 * (H*np.sin(theta)*cosdelta + np.cos(theta)*cosdelta*np.sin(H))