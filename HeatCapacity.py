from typing import Optional

import numpy as np
import numpy.typing as npt

floatarr = npt.NDArray[np.float64]


def ThermalTimeScale(C, T, I):
    return C * T / I


def C(
    f_o: float | floatarr, f_i: float | floatarr, T: Optional[float | floatarr] = None
) -> float | floatarr:
    """Calculate heat capacity from ocean and ice fractions
    f_o: fraction of planet which is ocean
    f_i: fraction of ocean which is ice

    returns: heat capacity, J m^-2 K^-1
    """
    # C = ρ c_p Δl
    # ρ - density
    # c_p - heat capacity
    # Δl - depth of atmosphere
    C_l = 5.25 * 10**6  # J m^-2 K^-1 # heat capacity over land
    ## North et al 1983 ##
    # Δl = 75m
    C_o = 60 * C_l  # heat capacity over ocean
    C_i = 9.2 * C_l  # heat capacity over ice
    # ## WK97 ##
    # # 263 K < T < 273 K
    # if 263 < T < 273:
    #     C_i = 9.2 * C_l
    # elif T < 263:
    #     C_i = 2.0 * C_l
    # else:
    #     C_i = ???
    # # Δl = 50m
    # C_o = 40 * C_l
    return (1 - f_o) * C_l + f_o * ((1 - f_i) * C_o + f_i * C_i)
