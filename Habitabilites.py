# SMS08 - 3.2 "fractional habitability"
# Habitable if temperature is between 273K and 373K (0-100C)
# Habitabilty function: H[a, λ, t] = {1 if 273 < t < 273 ; 0 otherwise}

# -> Temporal habitability: f_time[a, λ] = int_0^P(H[a,λ,t] dt) / P
# i.e. integrate over a period P to find if [a,λ] is habitable throughout

# -> Planetary habitability: f_area[a, t] = int_{-pi/2}^{pi/2}(H[a,λ,t] cosλ dλ) / 2
# i.e. integrate over all latitudes to find if [a, t] is habitable

# => combine both to get the net fractional habitability, weighted by area
# f_hab[a] = int_{-pi/2}^{pi/2}( int_0^P(H[a,λ,t] dt) cosλ dλ) / 2P
# fraction of λ-t plane which is habitable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from filemanagement import read_files


def H(temps: npt.NDArray) -> npt.NDArray:
    """temps: numpy array of temperatures.
    returns: 1 if 273 < temp < 373 ; 0 otherwise, for each temp in temps"""
    ret = np.ones_like(temps)
    ret[temps > 373] = 0
    ret[temps < 273] = 0
    return ret


def f_time(temps: npt.NDArray, times: npt.NDArray, dt: float) -> npt.NDArray:
    H_temps = H(temps) * dt
    H_temps_sum = np.sum(H_temps, axis=0) / (times[-1] - times[0])
    return H_temps_sum


def f_area(temps: npt.NDArray, lats: npt.NDArray, dlat: float) -> npt.NDArray:
    dx = np.cos(lats) * dlat
    H_temps = H(temps) * dx
    H_temps_sum = np.sum(H_temps, axis=1) / np.sum(dx)
    return H_temps_sum


def f_hab(
    temps: npt.NDArray, lats: npt.NDArray, dlat: float, times: npt.NDArray, dt: float
) -> float:
    dx = np.cos(lats) * dlat
    H_temps = H(temps) * dt * dx
    H_temps_sum = np.sum(H_temps, axis=1) / np.sum(dx)
    H_temps_sum_sum = np.sum(H_temps_sum, axis=0) / (times[-1] - times[0])
    return H_temps_sum_sum


if __name__ == "__main__":
    # times = np.linspace(0, 1000, 100)
    # temps = np.random.random(size=(100, 50)) * 10 + 268
    # lats = np.linspace(-np.pi / 2, np.pi / 2, 50)
    # dlat = abs(lats[1] - lats[0])

    # # print(f_time(temps, times, 1))
    # # print(f_area(temps, lats, dlat))
    # print(f_hab(temps, lats, dlat, times, 10))

    times, temps, degs = read_files(["InLat_1.npz"])
    times = times[0][365 * 150 : 365 * 151 + 1]
    temps = temps[0].T[365 * 150 : 365 * 151 + 1]
    degs = degs[0]
    lats = np.deg2rad(degs)

    dt = abs(times[1] - times[0])
    dlat = abs(degs[1] - degs[0])

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(degs, f_time(temps, times, dt))
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("f_time")
    ax2.plot(times, f_area(temps, lats, dlat))
    ax2.set_xlabel("Time")
    ax2.set_ylabel("f_area")
    print(f_hab(temps, lats, dlat, times, dt))

    plt.show()
