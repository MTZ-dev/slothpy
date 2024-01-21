import sys
from IPython import get_ipython
from slothpy.core._slothpy_exceptions import SltInputError
from slothpy._general_utilities._system import _is_notebook


class Settings:
    def __init__(self):
        self._settings = {
            "monitor": False,
            "traceback": False,
        }

    @property
    def monitor(self):
        return self._settings["monitor"]

    @monitor.setter
    def monitor(self, value):
        if isinstance(value, bool):
            self._settings["monitor"] = value
        else:
            raise SltInputError(
                ValueError("Monitor setter accepts only True or False.")
            )

    @property
    def traceback(self):
        return self._settings["traceback"]

    @traceback.setter
    def traceback(self, value):
        if isinstance(value, bool):
            self._settings["traceback"] = value
        else:
            raise SltInputError(
                ValueError("Traceback setter accepts only True or False.")
            )

    def show(self):
        """Print current settings for debugging purposes."""
        for name, value in self._settings.items():
            print(f"{name}: {value}")


settings = Settings()


def turn_on_monitor():
    """
    Run this to turn on the SlothPy_Monitor utility.
    """
    settings.monitor = True


def turn_off_monitor():
    """
    Run this to turn off the SlothPy_Monitor utility.
    """
    settings.monitor = False


# Set a custom traceback limit for printing the SltErrors
# for system and Jupyter Notebook. Edit it for debugging.
def set_plain_error_reporting_mode():
    """
    Run this after set_default_error_reporting_mode to return to the custom
    SlothPy-style error printing without tracebacks.
    """
    if _is_notebook():
        get_ipython().run_line_magic("xmode", "Plain")
    else:
        sys.tracebacklimit = 0

    settings.traceback = False


def set_default_error_reporting_mode():
    """
    Run this after the module import to return to the default full error
    tracebacks.
    """
    if _is_notebook():
        get_ipython().run_line_magic("xmode", "Context")
    else:
        sys.tracebacklimit = None

    settings.traceback = True
