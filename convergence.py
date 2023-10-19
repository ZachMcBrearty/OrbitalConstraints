import numpy as np

from OrbitalConstraints import climate_model_in_lat
from filemanagement import load_config

conf = load_config()


def convergence_test(times, temps, rtol=0.01):
    tq = temps.reshape(temps.shape[0], -1, 365)
    tqa = np.average(tq, axis=(0, 2))

    i = 0
    while not np.isclose(tqa[i], tqa[i + 1], rtol=rtol):
        i += 1
    return times[i]
