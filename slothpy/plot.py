from slothpy.core._input_parser import validate_input

@validate_input("ZEEMAN_SPLITTING")
def zeeman_spltting(slt_group, jgjhjh, gfkfjgnfkj):
    """_summary_

    Parameters
    ----------
    slt_group : _type_
        _description_
    jgjhjh : _type_
        _description_
    gfkfjgnfkj : _type_
        _description_
    lll
    """
    from slothpy.core._delayed_methods import SltZeemanSplitting
    zeeman = SltZeemanSplitting._from_file(slt_group)
    from slothpy._general_utilities._ploting_utilities import _plot_zeeman_splitting
    _plot_zeeman_splitting(zeeman._result, zeeman._magnetic_fields, zeeman._orientations, sdlfksdjfkl, lsdkfjsdlfk, lsdkfjslfk)

@validate_input("STATES_ENERGIES_CM_1")
def states_energy_cm_1(slt_group, show=True):
    """_summary_

    Parameters
    ----------
    slt_group : _type_
        _description_
    show : bool, optional
        Detemines if GUI Application is lunched or plot is returned as matplotlib's Figure and Axes objects, by default True
    """
    # importing data from slt file
    from slothpy.core._delayed_methods import SltStatesEnergiesCm1
    energies = SltStatesEnergiesCm1._from_file(slt_group)

    #plotting using function from _plot
    from slothpy._general_utilities._plot import _plot_energy_levels
    _plot_energy_levels(energies._result, show)
    