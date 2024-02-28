import inspect 
from functools import wraps
from numpy import array, float64
from slothpy.core._slothpy_exceptions import (SltInputError, SltSaveError)
from slothpy._general_utilities._grids_over_hemisphere import (
    lebedev_laikov_grid,
)
from slothpy._general_utilities._math_expresions import (
    _normalize_grid_vectors,
    _normalize_orientations,
    _normalize_orientation,
)
from slothpy._general_utilities._constants import (
    RED,
    GREEN,
    BLUE,
    PURPLE,
    YELLOW,
    RESET,
)
from slothpy._general_utilities._io import (
    _group_exists,
    _get_soc_energies_cm_1,
    _get_states_magnetic_momenta,
    _get_states_total_angular_momenta,
    _get_total_angular_momneta_matrix,
    _get_magnetic_momenta_matrix,
)


def validate_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        if "slt_save" in bound_args.arguments.keys() and bound_args.arguments["slt_save"] is not None:
            if _group_exists(bound_args.arguments["self"]._hdf5, bound_args.arguments["slt_save"]):
                raise SltSaveError(
                    bound_args.arguments["self"]._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + bound_args.arguments["slt_save"]
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            for name, value in bound_args.arguments.items():
                match name:
                    case "fields":
                        value = array(value, copy=False, order='C', dtype=float64)
                        if value.ndim != 1:
                            raise ValueError("The list of fields has to be a 1D array.")
                    case "temperatures":
                        value = array(value, copy=False, order='C', dtype=float64)
                        if value.ndim != 1:
                            raise ValueError("The list of temperatures has to be a 1D array.")
                    case "grid":
                        if isinstance(value, int):
                            value = lebedev_laikov_grid(value)
                        else:
                            value = _normalize_grid_vectors(value)
                bound_args.arguments[name] = value                    
        except Exception as exc:
            raise SltInputError(exc) from None

        return func(**bound_args.arguments)
    
    return wrapper

