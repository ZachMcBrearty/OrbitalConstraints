from Constants import *


## derivatives ##
def forwarddifference(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 1] - x[i]) / dx


def centraldifference(x: list[float] | floatarr, i: int, dx: float) -> float:
    # (x[i+1/2] - x[i-1/2]) / dx
    # let x[i+(-)1/2] = (x[i+(-)1] + x[i]) / 2
    # => (x[i+1] - x[i-1]) / (2*dx)
    return (x[i + 1] - x[i - 1]) / (2 * dx)


def backwarddifference(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i] - x[i - 1]) / dx


def forwardbackward_pole(T: floatarr, dx: float) -> float:
    """Used for pole at the start of the array, i.e. i = 0
    Assumes dT/dx = 0 at the pole"""
    # Forward: d^2T/dx^2 = (dT/dx|i=1 - dT/dx|i=0) / dx
    # dT/dx|i=0 == 0
    # Backward: d^2T/dx^2 = (T(i=1) - T(i=0)) / dx^2
    return (T[1] - T[0]) / dx**2


def backwardforward_pole(x: floatarr, dx: float) -> float:
    """Used for pole at the end of the array, i.e. i = len(x)-1
    Assumes dx/dt = 0 at the pole"""
    # Backward: d^2T/dx^2 = (dT/dx|i=i_max - dT/dx|i=i_max-1) / dx
    # dT/dx|i=i_max == 0
    # Forward: d^2T/dx^2 = -(T(i=i_max) - T(i=i_max-1)) / dx^2
    # => d^2T/dx^2 = (T(i=i_max-1) - T(i=i_max)) / dx^2
    # in python the final entry is -1 and last to final is -2
    return (x[-2] - x[-1]) / dx**2


def centralbackward_edge(x: floatarr, dx: float) -> float:
    """Used for one along from the start of the array, i.e. i = 1"""
    # Central: d^2T/dx^2 = (dT/dx|i=2 - dT/dx|i=0) / 2dx
    # dT/dx|i=0 == 0
    # Backward: d^2T/dx^2 = (T(i=2) - T(i=1)) / 2dx^2
    return (x[2] - x[1]) / (2 * dx**2)


def centralcentral_firstedge(x: floatarr, dx: float) -> float:
    """Used for one along from the start of the array, i.e. i = 1"""
    # Central: d^2T/dx^2 = (dT/dx|i=2 - dT/dx|i=0) / 2dx
    # dT/dx|i=0 == 0
    # Central: d^2T/dx^2 = (T(i=3) - T(i=1)) / 4dx^2
    return (x[3] - x[1]) / (4 * dx**2)


def centralcentral_secondedge(x: floatarr, dx: float) -> float:
    """Used for one along from the end of the array, i.e. i = len(x)-2"""
    # Central: d^2T/dx^2 = (dT/dx|i=i_max - dT/dx|i=i_max-2) / 2dx
    # dT/dx|i=i_max == 0
    # Central: d^2T/dx^2 = -(T(i=i_max-1) - T(i=i_max-3)) / 4dx^2
    return (x[-4] - x[-2]) / (4 * dx**2)


def centralforward_edge(x: floatarr, dx: float) -> float:
    """Used for one along from the end of the array, i.e. i = len(x)-2"""
    # Central: d^2T/dx^2 = (dT/dx|i=i_max - dT/dx|i=i_max-2) / 2dx
    # dT/dx|i=i_max == 0
    # Forward: d^2T/dx^2 = -(T(i=i_max-1) - T(i=i_max-2)) / 2dx^2
    # => d^2T/dx^2 = (T(i=i_max-2) - T(i=i_max-1)) / 2dx^2
    return (x[-3] - x[-2]) / (2 * dx**2)


def forward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i + 1] + x[i]) / dx**2


def central2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i + 2] - 2 * x[i] + x[i - 2]) / (2 * dx) ** 2


def backward2ndorder(x: list[float] | floatarr, i: int, dx: float) -> float:
    return (x[i] - 2 * x[i - 1] + x[i - 2]) / dx**2
