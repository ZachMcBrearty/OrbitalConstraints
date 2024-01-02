from Constants import *
import numpy as np


def r(pos):
    return np.sqrt(np.sum(pos**2))


def line(x, p1, p2):
    # y - y2 = m(x - x2); m = (y2-y1) / (x2 - x1)
    return p2[1] + (p2[1] - p1[1]) * (x - p2[0]) / (p2[0] - p1[0])


def check_eclipse(starpos, gaspos, moonpos, star_rad, gas_rad) -> float:
    # if the planet is infront of the gas giant do nothing
    # i.e. the planet cannot block the light for the moon
    if r(moonpos - starpos) < r(gaspos - starpos):
        return 1.0
    # find unit vector from star to gas giant
    star_to_gas_dir = (gaspos - starpos) / r(gaspos - starpos)
    # rotate 90deg left and right to get edges of the star and planet
    perp_up = np.array([-star_to_gas_dir[1], star_to_gas_dir[0]])
    perp_down = np.array([star_to_gas_dir[1], -star_to_gas_dir[0]])
    star_up = starpos + perp_up * star_rad
    star_down = starpos + perp_down * star_rad
    gas_up = gaspos + perp_up * gas_rad
    gas_down = gaspos + perp_down * gas_rad

    # logic is reversed since the system is "upside-down"
    if moonpos[0] < 0:
        # Umbra:
        # lines from edges of star to same edges to planet
        if (moonpos[1] > line(moonpos[0], star_up, gas_up)) and (
            moonpos[1] < line(moonpos[0], star_down, gas_down)
        ):
            q = 0.0
        # Penumbra:
        # lines from edges of star to opposite edges of planet
        elif (moonpos[1] > line(moonpos[0], star_down, gas_up)) and (
            moonpos[1] < line(moonpos[0], star_up, gas_down)
        ):
            q = 0.5
        # unblocked
        else:
            q = 1.0
    else:
        if (moonpos[1] < line(moonpos[0], star_up, gas_up)) and (
            moonpos[1] > line(moonpos[0], star_down, gas_down)
        ):
            q = 0.0
        elif (moonpos[1] < line(moonpos[0], star_down, gas_up)) and (
            moonpos[1] > line(moonpos[0], star_up, gas_down)
        ):
            q = 0.5
        else:
            q = 1.0
    return q


def r_explicit(x, y):
    return np.sqrt(x * x + y * y)


def line_explicit(x, p1_x, p1_y, p2_x, p2_y):
    # y - y2 = m(x - x2); m = (y2-y1) / (x2 - x1)
    return p2_y + (p2_y - p1_y) * (x - p2_x) / (p2_x - p1_x)


def check_eclipse_explicit(
    starpos: tuple[float, float],
    gaspos: tuple[float, float],
    moonpos: tuple[float, float],
    star_rad: float,
    gas_rad: float,
) -> float:
    # if the planet is infront of the gas giant do nothing
    # i.e. the planet cannot block the light for the moon
    gas_to_star_dist = r_explicit(gaspos[0] - starpos[0], gaspos[1] - starpos[1])
    if r_explicit(moonpos[0] - starpos[0], moonpos[1] - starpos[1]) < gas_to_star_dist:
        return 1.0
    # find unit vector from star to gas giant
    star_to_gas_dir_x = (gaspos[0] - starpos[0]) / gas_to_star_dist
    star_to_gas_dir_y = (gaspos[1] - starpos[1]) / gas_to_star_dist
    # rotate 90deg left and right to get edges of the star and planet
    perp_up_x = -star_to_gas_dir_y
    perp_up_y = star_to_gas_dir_x
    perp_down_x = star_to_gas_dir_y
    perp_down_y = -star_to_gas_dir_x

    star_up_x = starpos[0] + perp_up_x * star_rad
    star_up_y = starpos[1] + perp_up_y * star_rad
    star_down_x = starpos[0] + perp_down_x * star_rad
    star_down_y = starpos[1] + perp_down_y * star_rad

    gas_up_x = gaspos[0] + perp_up_x * gas_rad
    gas_up_y = gaspos[1] + perp_up_y * gas_rad
    gas_down_x = gaspos[0] + perp_down_x * gas_rad
    gas_down_y = gaspos[1] + perp_down_y * gas_rad

    # logic is reversed since the system is "upside-down"
    if moonpos[0] < 0:
        # Umbra:
        # lines from edges of star to same edges to planet
        if (
            moonpos[1]
            > line_explicit(moonpos[0], star_up_x, star_up_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            < line_explicit(
                moonpos[0], star_down_x, star_down_y, gas_down_x, gas_down_y
            )
        ):
            q = 0.0
        # Penumbra:
        # lines from edges of star to opposite edges of planet
        elif (
            moonpos[1]
            > line_explicit(moonpos[0], star_down_x, star_down_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            < line_explicit(moonpos[0], star_up_x, star_up_y, gas_down_x, gas_down_y)
        ):
            q = 0.5
        # unblocked
        else:
            q = 1.0
    else:
        if (
            moonpos[1]
            < line_explicit(moonpos[0], star_up_x, star_up_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            > line_explicit(
                moonpos[0], star_down_x, star_down_y, gas_down_x, gas_down_y
            )
        ):
            q = 0.0
        elif (
            moonpos[1]
            < line_explicit(moonpos[0], star_down_x, star_down_y, gas_up_x, gas_up_y)
        ) and (
            moonpos[1]
            > line_explicit(moonpos[0], star_up_x, star_up_y, gas_down_x, gas_down_y)
        ):
            q = 0.5
        else:
            q = 1.0
    return q


def orbital_model(conf: CONF_PARSER_TYPE, dt_steps=10):
    G_prime = G * (24 * 60 * 60) ** 2  # s^-2 -> days^-2
    star = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
    gas = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
    moon = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)

    dt = conf.getfloat("PDE", "timestep") / dt_steps  # days
    M_star = conf.getfloat("ORBIT", "starmass") * MASS["solar"]
    r_star = conf.getfloat("ORBIT", "starradius") * RADIUS["solar"]
    M_gas = conf.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    r_gas = conf.getfloat("ORBIT", "gasgiantradius") * RADIUS["jupiter"]

    dens_moon = conf.getfloat("ORBIT", "moondensity")
    r_moon = conf.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    M_moon = 4 * np.pi / 3 * r_moon**3 * dens_moon

    gas_a = conf.getfloat("ORBIT", "gassemimajoraxis") * AU
    gas_ecc = conf.getfloat("ORBIT", "gaseccentricity")
    moon_a = conf.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = conf.getfloat("ORBIT", "mooneccentricity")

    gas[0][0] = gas_a * (1 + gas_ecc)
    moon[0][0] = gas[0][0] + moon_a * (1 + moon_ecc)

    # v^2 = GM(2/r - 1/a) = GM/a
    # use vis-viva equation to put giant (and moon) on orbit around the star
    gas[1][1] = np.sqrt(G_prime * M_star * (2 / gas[0][0] - 1 / gas_a))
    # then use vis-viva to put moon on orbit around giant
    moon[1][1] = gas[1][1] + np.sqrt(
        G_prime * M_gas * (2 / (moon_a * (1 + moon_ecc)) - 1 / moon_a)
    )
    # then use conservation of momentum to set the star on the opposite orbit
    # so center of momentum doesnt move
    star[1][1] = -gas[1][1] * M_gas / M_star - moon[1][1] * M_moon / M_star  #
    epss = 0
    # setup Leapfrog (ie put velocities back 1/2 timestep)
    stargas = np.sum((gas[0] - star[0]) ** 2 + epss) ** (-3 / 2)
    starmoon = np.sum((moon[0] - star[0]) ** 2 + epss) ** (-3 / 2)
    moongas = np.sum((gas[0] - moon[0]) ** 2 + epss) ** (-3 / 2)

    star[2] = -(
        G_prime * M_gas * (star[0] - gas[0]) * stargas
        + G_prime * M_moon * (star[0] - moon[0]) * starmoon
    )
    gas[2] = -(
        +G_prime * M_star * (gas[0] - star[0]) * stargas
        + G_prime * M_moon * (gas[0] - moon[0]) * moongas
    )
    moon[2] = -(
        +G_prime * M_gas * (moon[0] - gas[0]) * moongas
        + G_prime * M_star * (moon[0] - star[0]) * starmoon
    )
    leapfrog_update_matrix = np.array([[1, 0, 0], [0, 1, -dt / 2], [0, 0, 0]])
    star = leapfrog_update_matrix @ star
    gas = leapfrog_update_matrix @ gas
    moon = leapfrog_update_matrix @ moon

    update_matrix = np.array([[1, dt, dt**2], [0, 1, dt], [0, 0, 0]])
    # eclipsed = check_eclipse(star[0], gas[0], moon[0], r_star, r_gas)
    eclipsed = 0

    yield star[0], gas[0], moon[0], eclipsed
    while True:
        stargas = np.sum((gas[0] - star[0]) ** 2 + epss) ** (-3 / 2)
        starmoon = np.sum((moon[0] - star[0]) ** 2 + epss) ** (-3 / 2)
        moongas = np.sum((gas[0] - moon[0]) ** 2 + epss) ** (-3 / 2)

        star[2] = -(
            G_prime * M_gas * (star[0] - gas[0]) * stargas
            + G_prime * M_moon * (star[0] - moon[0]) * starmoon
        )
        gas[2] = -(
            +G_prime * M_star * (gas[0] - star[0]) * stargas
            + G_prime * M_moon * (gas[0] - moon[0]) * moongas
        )
        moon[2] = -(
            +G_prime * M_gas * (moon[0] - gas[0]) * moongas
            + G_prime * M_star * (moon[0] - star[0]) * starmoon
        )

        star = update_matrix @ star
        gas = update_matrix @ gas
        moon = update_matrix @ moon
        # eclipsed = check_eclipse(star[0], gas[0], moon[0], r_star, r_gas)
        yield star[0], gas[0], moon[0], eclipsed


def orbital_model_explicit(conf: CONF_PARSER_TYPE, dt_steps=10):
    G_prime = G * (24 * 60 * 60) ** 2  # s^-2 -> days^-2
    star_x, star_y, star_vx, star_vy, star_ax, star_ay = 0, 0, 0, 0, 0, 0
    gas_x, gas_y, gas_vx, gas_vy, gas_ax, gas_ay = 0, 0, 0, 0, 0, 0
    moon_x, moon_y, moon_vx, moon_vy, moon_ax, moon_ay = 0, 0, 0, 0, 0, 0

    dt = conf.getfloat("PDE", "timestep") / dt_steps  # days
    M_star = conf.getfloat("ORBIT", "starmass") * MASS["solar"]
    r_star = conf.getfloat("ORBIT", "starradius") * RADIUS["solar"]
    M_gas = conf.getfloat("ORBIT", "gasgiantmass") * MASS["jupiter"]
    r_gas = conf.getfloat("ORBIT", "gasgiantradius") * RADIUS["jupiter"]

    dens_moon = conf.getfloat("ORBIT", "moondensity")
    r_moon = conf.getfloat("ORBIT", "moonradius") * RADIUS["luna"]
    M_moon = 4 * np.pi / 3 * r_moon**3 * dens_moon

    gas_a = conf.getfloat("ORBIT", "gassemimajoraxis") * AU
    gas_ecc = conf.getfloat("ORBIT", "gaseccentricity")
    moon_a = conf.getfloat("ORBIT", "moonsemimajoraxis") * AU
    moon_ecc = conf.getfloat("ORBIT", "mooneccentricity")

    gas_x = gas_a * (1 + gas_ecc)
    moon_x = gas_x + moon_a * (1 + moon_ecc)

    # v^2 = GM(2/r - 1/a) = GM/a
    # use vis-viva equation to put giant (and moon) on orbit around the star
    gas_vy = np.sqrt(G_prime * M_star * (2 / gas_x - 1 / gas_a))
    # then use vis-viva to put moon on orbit around giant
    moon_vy = gas_vy + np.sqrt(
        G_prime * M_gas * (2 / (moon_a * (1 + moon_ecc)) - 1 / moon_a)
    )
    # then use conservation of momentum to set the star on the opposite orbit
    # so center of momentum doesnt move
    star_vy = -gas_vy * M_gas / M_star - moon_vy * M_moon / M_star  #
    # setup Leapfrog (ie put velocities back 1/2 timestep)
    stargas = ((gas_x - star_x) ** 2 + (gas_y - star_y) ** 2) ** (-3 / 2)
    starmoon = ((moon_x - star_x) ** 2 + (moon_y - star_y) ** 2) ** (-3 / 2)
    moongas = ((moon_x - gas_x) ** 2 + (moon_y - gas_y) ** 2) ** (-3 / 2)

    star_ax = -(
        G_prime * M_gas * (star_x - gas_x) * stargas
        + G_prime * M_moon * (star_x - moon_x) * starmoon
    )
    star_ay = -(
        G_prime * M_gas * (star_y - gas_y) * stargas
        + G_prime * M_moon * (star_y - moon_y) * starmoon
    )
    gas_ax = -(
        G_prime * M_star * (gas_x - star_x) * stargas
        + G_prime * M_moon * (gas_x - moon_x) * moongas
    )
    gas_ay = -(
        G_prime * M_gas * (gas_y - star_y) * stargas
        + G_prime * M_moon * (gas_y - moon_y) * moongas
    )
    moon_ax = -(
        G_prime * M_gas * (moon_x - gas_x) * moongas
        + G_prime * M_star * (moon_x - star_x) * starmoon
    )
    moon_ay = -(
        G_prime * M_gas * (moon_y - gas_y) * moongas
        + G_prime * M_star * (moon_y - star_y) * starmoon
    )
    star_vx -= dt / 2 * star_ax
    star_vy -= dt / 2 * star_ay
    gas_vx -= dt / 2 * gas_ax
    gas_vy -= dt / 2 * gas_ay
    moon_vx -= dt / 2 * moon_ax
    moon_vy -= dt / 2 * moon_ay

    # update_matrix = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 0]])
    eclipsed = check_eclipse_explicit(
        (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), r_star, r_gas
    )

    # eclipsed = 0

    yield (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), eclipsed
    while True:
        stargas = ((gas_x - star_x) ** 2 + (gas_y - star_y) ** 2) ** (-3 / 2)
        starmoon = ((moon_x - star_x) ** 2 + (moon_y - star_y) ** 2) ** (-3 / 2)
        moongas = ((moon_x - gas_x) ** 2 + (moon_y - gas_y) ** 2) ** (-3 / 2)
        star_ax = -(
            G_prime * M_gas * (star_x - gas_x) * stargas
            + G_prime * M_moon * (star_x - moon_x) * starmoon
        )
        star_ay = -(
            G_prime * M_gas * (star_y - gas_y) * stargas
            + G_prime * M_moon * (star_y - moon_y) * starmoon
        )
        gas_ax = -(
            G_prime * M_star * (gas_x - star_x) * stargas
            + G_prime * M_moon * (gas_x - moon_x) * moongas
        )
        gas_ay = -(
            G_prime * M_gas * (gas_y - star_y) * stargas
            + G_prime * M_moon * (gas_y - moon_y) * moongas
        )
        moon_ax = -(
            G_prime * M_gas * (moon_x - gas_x) * moongas
            + G_prime * M_star * (moon_x - star_x) * starmoon
        )
        moon_ay = -(
            G_prime * M_gas * (moon_y - gas_y) * moongas
            + G_prime * M_star * (moon_y - star_y) * starmoon
        )

        star_x += dt * star_vx + dt**2 * star_ax
        star_y += dt * star_vy + dt**2 * star_ay
        gas_x += dt * gas_vx + dt**2 * gas_ax
        gas_y += dt * gas_vy + dt**2 * gas_ay
        moon_x += dt * moon_vx + dt**2 * moon_ax
        moon_y += dt * moon_vy + dt**2 * moon_ay

        star_vx += dt * star_ax
        star_vy += dt * star_ay
        gas_vx += dt * gas_ax
        gas_vy += dt * gas_ay
        moon_vx += dt * moon_ax
        moon_vy += dt * moon_ay

        eclipsed = check_eclipse_explicit(
            (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), r_star, r_gas
        )
        yield (star_x, star_y), (gas_x, gas_y), (moon_x, moon_y), eclipsed
