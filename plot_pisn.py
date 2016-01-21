
"""
Calculate odd/even abundances for all stars and look for PISN signature.
"""
import matplotlib
matplotlib.rcParams["text.usetex"] = True

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from six import string_types

np.random.seed(123)


def atomic_number(element):
    """ Converts a string representation of an element and its ionization state
    to a floating point """
    
    periodic_table = """H                                                  He
                        Li Be                               B  C  N  O  F  Ne
                        Na Mg                               Al Si P  S  Cl Ar
                        K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                        Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                        Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                        Fr Ra Lr""" 
    
    lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
    actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"
    
    periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
        .replace(" Ra ", " Ra " + actinoids + " ").split()
    del actinoids, lanthanoids
    
    element = element.title()
    if not isinstance(element, string_types):
        raise TypeError("element must be represented by a string-type")
    
    return 1 + periodic_table.index(element)


def plot_atomic_number_abundances(apogee_rows, 
    comparison_stars=None, N_comparisons=100, wrt="H", cmap="terrain", 
    text_offset=None, scatter_kwargs=None, line_kwargs=None, **kwargs):
    """
    Plot the atomic number abundances for a given APOGEE ID.
    """


    wrt = wrt.upper()
    if wrt not in ("H", "FE"):
        raise ValueError("must be with respect to H or Fe")

    representative_uncertainty = {
        "C":  0.041,
        "N":  0.044,
        "O":  0.037,
        "Na": 0.111,
        "Mg": 0.032,
        "Al": 0.055,
        "Si": 0.041,
        "S":  0.054,
        "K":  0.069,
        "Ca": 0.043,
        "Ti": 0.072,
        "V":  0.146,
        "Mn": 0.041,
        "Fe": 0.019,
        "Ni": 0.034
    }

    line_kwargs = line_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}

    try:
        apogee_rows.data.dtype.names
    except:
        None
    else:
        apogee_rows = [apogee_rows]

    fig, ax = plt.subplots()

    # What elements do we have?
    elements = np.array([_.split("_")[0] for _ in apogee_rows[0].dtype.names \
        if _.endswith("_H") and len(_) < 5])
    N_elements, N_rows = len(elements), len(apogee_rows)

    cmap = plt.cm.get_cmap(cmap, N_rows)

    # Prepare the arrays.
    Z = np.array([atomic_number(element) for element in elements], dtype=int)
    abundances = np.nan * np.ones((N_rows, N_elements))

    # Sort the elements by atomic number.
    sort_indices = np.argsort(Z)
    Z, elements = Z[sort_indices], elements[sort_indices]

    uncertainties = np.array([representative_uncertainty[element.title()] \
        for element in elements])
    for i, apogee_row in enumerate(apogee_rows):

        offset = 0 if wrt == "H" else apogee_row["FE_H"]
        for j, element in enumerate(elements):
            abundances[i, j] = apogee_row["{0}_H".format(element)] - offset

        # Scatter dem points.
        kwds = {
            "facecolor": cmap(i),
            "s": 75,
            "label": r"${\rm %s}$" % apogee_row["APOGEE_ID"],
            "lw": 1.5
        }
        kwds.update(scatter_kwargs)
        ax.scatter(Z, abundances[i], **kwds)

        errorbar_kwds = {
            "fmt": None,
            "c": cmap(i),
            "ecolor": cmap(i),
            "elinewidth": 2,
            "zorder": -1
        }
        ax.errorbar(Z, abundances[i], yerr=uncertainties, **errorbar_kwds)

        # Draw lines for sequential atomic abundances.
        kwds = {
            "linestyle": "-",
            "zorder": -1,
            "c": cmap(i),
            "lw": 1.5,
        }
        kwds.update(line_kwargs)
        for index in np.where(np.diff(Z) == 1)[0]:
            ax.plot(Z[index:2 + index], abundances[i, index:2 + index], **kwds)

    if comparison_stars is not None:
        comparison_indices = np.random.randint(0, len(comparison_stars),
            N_comparisons)

        for i, comparison_row in enumerate(comparison_stars[comparison_indices]):

            if comparison_row["APOGEE_ID"] in list(apogee_rows["APOGEE_ID"]):
                continue

            offset = 0 if wrt == "H" else comparison_row["FE_H"]
            abundances = np.ones_like(Z, dtype=float)
            for j, element in enumerate(elements):
                abundances[j] = comparison_row["{0}_H".format(element)] - offset

            # Draw lines for sequential atomic abundances.
            kwds = {
                "linestyle": "-",
                "zorder": -1,
                "c": "k",
                "alpha": 0.05,
                "lw": 1,
                "zorder": -30,
            }
            for index in np.where(np.diff(Z) == 1)[0]:
                ax.plot(Z[index:2 + index], abundances[index:2 + index], **kwds)

    default_ypos = ax.get_ylim()[0] + 0.95 * np.ptp(ax.get_ylim())
    for j, element in enumerate(elements):
        if Z[j] % 2 > 0:
            sign, func = -1, np.min
        else:
            sign, func = +1, np.max

        if text_offset is None:
            ypos = default_ypos
        else:
            ypos = func(abundances[:, j]) + sign \
                * (text_offset + representative_uncertainty[element.title()])

        ax.text(Z[j], ypos, r"${\rm %s}$" % element.title(), 
            horizontalalignment="center", verticalalignment="center")

    # Labels, ticks and looks.
    ax.set_xlabel(r"${\rm Atomic}$ ${\rm Number},$ $Z$")
    if wrt == "H":
        ax.set_ylabel(r"${\rm [X/H]}$")
    else:
        ax.set_ylabel(r"${\rm [X/Fe]}$")

    ax.axhline(0, c="#666666", zorder=-100, linestyle=":")

    ax.set_xticks(Z)
    ax.yaxis.set_major_locator(MaxNLocator(6))

    ax.set_xlim(Z[0] - 1, Z[-1] + 1)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    return fig


def select_comparison_star(pisn_candidate, apogee_catalog):
    """
    Return a comparison (normal) star with similar stellar parameters,
    but normal [odd/even] ratios.
    """

    # Scaled L2.
    D = \
            ((pisn_candidate["TEFF"] - apogee_catalog["TEFF"])/np.ptp(apogee_catalog["TEFF"]))**2 \
        +   ((pisn_candidate["LOGG"] - apogee_catalog["LOGG"])/np.ptp(apogee_catalog["LOGG"]))**2 \
        +   ((pisn_candidate["FE_H"] - apogee_catalog["FE_H"])/np.ptp(apogee_catalog["FE_H"]))**2 \
        +   ((apogee_catalog["even_odd"] - 0)/np.ptp(apogee_catalog["even_odd"]))**2

    D[apogee_catalog["APOGEE_ID"] == pisn_candidate["APOGEE_ID"]] = np.inf

    index = np.nanargmin(D)
    return (apogee_catalog[index], index)



def plot_pisn_candidate_spectra(dispersion, pisn_candidate, comparison_star):
    """
    Plot a comparison star.
    """

    fig, ax = plt.subplots(1, 3) # 3 spectra 
    raise a




def misc_plots():


    # HRD
    fig, ax = plt.subplots()
    ax.scatter(apogee_rows["TEFF"], apogee_rows["LOGG"], c=colors)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])


    # Positions
    fig, ax = plt.subplots()
    ax.scatter(apogee_rows["RA"], apogee_rows["DEC"], c=colors)
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")


    fig, ax = plt.subplots()
    ax.scatter(apogee_rows["RA"], apogee_rows["VHELIO_AVG"], c=colors)
    ax.set_xlabel("RA")
    ax.set_ylabel("VRAD")

    # Find stars that are almost identical (in TEFF, LOGG, and FEH) to each of
    # our stars, then plot a completely normal (as having [odd/even] ~ 0)
    # spectrum and the spectrum for this star.

    # Highlight wavelengths (from APOGEE line list) where odd elements are
    # and where even elements are (in different colours)

    # Show the difference of those two spectra, and highlight areas where odd/even
    # elements are! YES!

    # ... then what? the rest is interpretation and positional info etc.


    raise a

if __name__ == "__main__":

    try:
        data
    except:
        from astropy.table import Table

        data = Table.read("../apogee/results-unregularized/results-unregularized-matched.fits")
        data = data[data["physical"]]


    elements = [_.split("_")[0].title() for _ in data.dtype.names \
        if _.endswith("_H") and len(_) < 5]

    odd_elements = [_.upper() for _ in elements if (atomic_number(_) % 2) > 0]
    even_elements = [_.upper() for _ in elements if (atomic_number(_) % 2) == 0]

    # Calculate odd/even abundances.
    odd_abundances = np.array([data["{}_H".format(element)] \
        for element in odd_elements])
    even_abundances = np.array([data["{}_H".format(element)] \
        for element in even_elements])

    # Calculate even/odd ratios.
    odd_abundance = np.mean(odd_abundances, axis=0)
    even_abundance = np.mean(even_abundances, axis=0)
    
    even_to_odd_ratio = even_abundance - odd_abundance
    data["even_odd"] = even_to_odd_ratio
    #data.write("results-unregularized-matched-even+odd.fits")
    

    # Ignore anything with [Fe/H] < -0.7 for a second.
    #even_to_odd_ratio[data["FE_H"] < -0.7] = -np.inf
    even_to_odd_ratio[data["FE_H"] > -0.23] = -np.inf


    N = 3
    cmap = plt.cm.get_cmap("Set1", N)

    colors = [cmap(i) for i in range(N)]
    y = even_to_odd_ratio + 0.2 * data["FE_H"]
    sort_indices = np.argsort(y)[::-1]


    fig, ax = plt.subplots()
    ax.scatter(data["FE_H"], y, facecolor="#666666")
    ax.scatter(
        data["FE_H"][sort_indices[:N]],
        y[sort_indices[:N]],
        c=colors, s=100)

    plot_atomic_number_abundances(data[sort_indices[:N]], wrt="Fe", 
        cmap="Set2",
        comparison_stars=data, N_comparisons=100)

    apogee_ids = []
    comparison_star_indices = []
    for pisn_candidate in data[sort_indices[:N]]:
        comparison_star, index = select_comparison_star(pisn_candidate, data)

        print("--")
        for star in (pisn_candidate, comparison_star):
            print("{0}: {1:.0f} / {2:.2f} / {3:.2f} / {4:.2f}".format(
                star["APOGEE_ID"], star["TEFF"], star["LOGG"], star["FE_H"],
                star["even_odd"]))
        print("--")
        apogee_ids.append(pisn_candidate["APOGEE_ID"])
        apogee_ids.append(comparison_star["APOGEE_ID"])

        comparison_star_indices.append(index)

    comparison_star_indices = np.array(comparison_star_indices)
    plot_atomic_number_abundances(data[comparison_star_indices], wrt="Fe", 
        cmap="Set2",
        comparison_stars=data, N_comparisons=100)


    raise a



# [ ] spectra for APOGEE_ids
# [ ] APOGEE Line list paper +/- Verne smith paper



    
