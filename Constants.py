from numpy import floating
from numpy.typing import NDArray

floatarr = NDArray[floating]

yeartosecond = 365.25 * 24 * 3600  # s / yr

spacedim_unit = None
timestep_unit = "days"
temp_unit = "K"
omega_unit = "days$^{-1}$"
obliquity_unit = r"$^{\circ}$"
a_unit = "au"
e_unit = None
