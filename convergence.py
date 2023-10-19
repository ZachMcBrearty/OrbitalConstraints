import numpy as np

from ClimateModel import climate_model_in_lat
from filemanagement import load_config

conf = load_config()


def convergence_test(temps, rtol=0.001, atol=0):
    """returns: how long the data set took to converge in years"""
    tq = temps[:, 1:].reshape(temps.shape[0], -1, 365)
    tqa = np.average(tq, axis=2)
    tqaa = np.average(tqa, axis=0)

    i = 0
    imax = len(tqaa) - 2
    while not np.isclose(tqaa[i], tqaa[i + 1], rtol=rtol, atol=0) and i != imax:
        i += 1
    if i == imax:
        return -1
    else:
        return i


conf.set("FILEMANAGEMENT", "save", "False")
conf.set("PDE", "time", "200")
conf.set("ORBIT", "a", "0.7")
degs, temps, times = climate_model_in_lat(conf)

print(convergence_test(temps, 0.0001))
