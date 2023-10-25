import numpy as np
import matplotlib.pyplot as plt

from ClimateModel import climate_model_in_lat
from filemanagement import load_config


def convergence_test(temps, rtol=0.001, atol=0, year_avg=1):
    """returns: how long the data set took to converge in years, -1 if never"""
    tq = temps[:, 1:].reshape(temps.shape[0], -1, 365 * year_avg)
    tqa = np.average(tq, axis=2)  # average over time
    tqaa = np.average(
        tqa, axis=0, weights=np.cos(np.linspace(-np.pi / 2, np.pi / 2, temps.shape[0]))
    )  # average over latitude, weighted by area

    i = 0
    imax = len(tqaa) - 2
    while not np.isclose(tqaa[i], tqaa[i + 1], rtol=rtol, atol=atol) and i != imax:
        i += 1
    if i == imax:
        return -1, 0
    else:
        return i * year_avg, tqaa[i]


def test_a_convergence(conf, a_min, a_max, a_step, rtol=0.001, plot=False):
    tests = []
    convtemps = []
    as_ = np.arange(a_min, a_max, a_step)
    for a in as_:
        print(f"Running a={a}")
        conf.set("ORBIT", "a", str(a))
        degs, temps, times = climate_model_in_lat(conf)
        t, temp = convergence_test(temps, rtol, year_avg=1)
        tests.append(t)
        convtemps.append(temp)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(as_, tests)
        ax2.scatter(as_, convtemps)
        ax2.set_xlabel("Semi major axis, a, AU")
        ax1.set_xticks(np.linspace(min(as_), max(as_), 10))
        ax2.set_xticks(np.linspace(min(as_), max(as_), 10))
        ax1.set_ylabel("Time to converge, years")
        ax2.set_ylabel("Global convergent temperature, K")
        plt.show()

    return as_, tests


def test_e_convergence(conf, e_min, e_max, e_step, rtol=0.001, plot=False):
    tests = []
    convtemps = []
    es_ = np.arange(e_min, e_max, e_step)
    for e in es_:
        print(f"Running e={e}")
        conf.set("ORBIT", "e", str(e))
        degs, temps, times = climate_model_in_lat(conf)
        t, temp = convergence_test(temps, rtol, year_avg=1)
        tests.append(t)
        convtemps.append(temp)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(es_, tests)
        ax2.scatter(es_, convtemps)
        ax2.set_xlabel("Eccentricity, e")
        ax1.set_xticks(np.linspace(min(es_), max(es_), 10))
        ax2.set_xticks(np.linspace(min(es_), max(es_), 10))
        ax1.set_ylabel("Time to converge, years")
        ax2.set_ylabel("Global convergent temperature, K")
        plt.show()

    return es_, tests


def test_delta_convergence(conf, d_min, d_max, d_step, rtol=0.001, plot=False):
    """obliquity testing in degrees"""
    tests = []
    convtemps = []
    ds_ = np.arange(d_min, d_max, d_step)
    for d in ds_:
        print(f"Running delta={d}")
        conf.set("PLANET", "obliquity", str(d))
        degs, temps, times = climate_model_in_lat(conf)
        t, temp = convergence_test(temps, rtol, year_avg=1)
        tests.append(t)
        convtemps.append(temp)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(ds_, tests)
        ax2.scatter(ds_, convtemps)
        ax2.set_xlabel(r"Obliquity, $\delta$, degrees")
        ax1.set_xticks(np.linspace(min(ds_), max(ds_), 10))
        ax2.set_xticks(np.linspace(min(ds_), max(ds_), 10))
        ax1.set_ylabel("Time to converge, years")
        ax2.set_ylabel("Global convergent temperature, K")
        plt.show()

    return ds_, tests


def test_omega_convergence(conf, o_min, o_max, o_step, rtol=0.001, plot=False):
    """rotation speed testing, per day"""
    tests = []
    convtemps = []
    os_ = np.arange(o_min, o_max, o_step)
    for o in os_:
        print(f"Running omega={o}")
        conf.set("PLANET", "Omega", str(o))
        degs, temps, times = climate_model_in_lat(conf)
        t, temp = convergence_test(temps, rtol, year_avg=1)
        tests.append(t)
        convtemps.append(temp)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(os_, tests)
        ax2.scatter(os_, convtemps)
        ax2.set_xlabel(r"$\Omega$, days$^{-1}$")
        ax1.set_xticks(np.linspace(min(os_), max(os_), 10))
        ax2.set_xticks(np.linspace(min(os_), max(os_), 10))
        ax1.set_ylabel("Time to converge, years")
        ax2.set_ylabel("Global convergent temperature, K")
        plt.show()

    return os_, tests


def test_temp_convergence(conf, t_min, t_max, t_step, rtol=0.001, plot=False):
    """initial temperature testing"""
    tests = []
    convtemps = []
    ts_ = np.arange(t_min, t_max, t_step)
    for t in ts_:
        print(f"Running inittemp={t}")
        conf.set("PDE", "start_temp", str(t))
        degs, temps, times = climate_model_in_lat(conf)
        t, temp = convergence_test(temps, rtol, year_avg=1)
        tests.append(t)
        convtemps.append(temp)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(ts_, tests)
        ax2.scatter(ts_, convtemps)
        ax2.set_xlabel(r"Initial Temperature, K")
        ax1.set_xticks(np.linspace(min(ts_), max(ts_), 10))
        ax2.set_xticks(np.linspace(min(ts_), max(ts_), 10))
        ax1.set_ylabel("Time to converge, years")
        ax2.set_ylabel("Global convergent temperature, K")
        plt.show()

    return ts_, tests


if __name__ == "__main__":
    conf = load_config()

    conf.set("FILEMANAGEMENT", "save", "False")
    conf.set("FILEMANAGEMENT", "plot", "False")

    conf.set("PDE", "spacedim", "60")
    conf.set("PDE", "time", "100")
    conf.set("PDE", "timestep", "1")
    conf.set("PDE", "start_temp", "350")

    conf.set("PLANET", "omega", "1")
    conf.set("PLANET", "land_frac_type", "uniform:0.7")
    conf.set("PLANET", "obliquity", "23.5")

    conf.set("ORBIT", "a", "1")
    conf.set("ORBIT", "e", "0")

    # print(test_a_convergence(conf, 0.5, 2.05, 0.1, rtol=0.0001, plot=True))
    # print(test_e_convergence(conf, 0, 0.91, 0.1, rtol=0.0001, plot=True))
    # print(test_delta_convergence(conf, 0, 181, 10, rtol=0.0001, plot=True))
    # print(test_omega_convergence(conf, 0.3, 3, 0.3, rtol=0.0001, plot=True))
    print(test_temp_convergence(conf, 100, 501, 50, rtol=0.0001, plot=True))
    # degs, temps, times = climate_model_in_lat(conf)

    # print(convergence_test(temps, 0.0001))
