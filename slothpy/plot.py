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
    """
    from slothpy.core._delayed_methods import SltZeemanSplitting
    zeeman = SltZeemanSplitting._from_file(slt_group)
    from slothpy._general_utilities._ploting_utilities import _plot_zeeman_splitting
    _plot_zeeman_splitting(zeeman._result, zeeman._magnetic_fields, zeeman._orientations, sdlfksdjfkl, lsdkfjsdlfk, lsdkfjslfk)

    