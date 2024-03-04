import sys
from importlib import resources, resources as pkg_resources
from configparser import ConfigParser
from importlib import resources
from typing import Literal
from IPython import get_ipython
from slothpy.core._slothpy_exceptions import SltInputError
from slothpy._general_utilities._system import _is_notebook
from slothpy._general_utilities._constants import YELLOW, BLUE, GREEN, RESET


class SltSettings:

    def __init__(self):
        self._settings = {
            "monitor": False,
            "traceback": False,
            "log_level": 0,
            "precision": "double",
        }
        # settings_path = join("static", "settings.ini")
        config = ConfigParser()
        with resources.open_text("slothpy", "settings.ini") as file:
            config.read_file(file)

        # Update boolean settings
        for key in ["monitor", "traceback"]:
            if key in config['DEFAULT']:
                self._settings[key] = config['DEFAULT'].getboolean(key)
        
        # Update string setting
        if "precision" in config['DEFAULT']:
            self._settings["precision"] = config['DEFAULT'].get("precision")
        
        # Update integer setting
        if "log_level" in config['DEFAULT']:
            self._settings["log_level"] = config['DEFAULT'].getint("log_level")

    def __repr__(self):
        return f"<SltSettings object.>"

    def __str__(self):
        representation = BLUE + "Sloth" + YELLOW + "Py" + RESET + " Settings:\n"
        for name, value in self._settings.items():
            representation += GREEN + f"{name}" + RESET + f": {value}\n"
        return representation.strip()

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
        if self._settings["traceback"]:
            _set_default_error_reporting_mode()
        else:
            _set_plain_error_reporting_mode()

    @property
    def precision(self):
        return self._settings["precision"]

    @precision.setter
    def precision(self, value: Literal["double", "single"]):
        if value in ["double", "single"]:
            self._settings["precision"] = value
        else:
            raise SltInputError(
                ValueError("Precision setter accepts only 'double' or 'single' literals.")
            )
    
    @property
    def log_level(self):
        return self._settings["log_level"]

    @log_level.setter
    def log_level(self, value):
        if isinstance(value, int):
            self._settings["log_level"] = value
        else:
            raise SltInputError(
                ValueError("Log level setter accepts only integers.")
            )

    def save_settings(self):
        """Write the current settings back to the configuration file."""
        config = ConfigParser()
        config['DEFAULT'] = self._settings
        
        config_file_path = pkg_resources.files('slothpy') / 'settings.ini'
        
        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    def show(self):
        """Print current settings for debugging purposes."""
        print(self.__str__())


settings = SltSettings()


def _set_plain_error_reporting_mode():
    if _is_notebook():
        get_ipython().run_line_magic("xmode", "Plain")
    else:
        sys.tracebacklimit = 0


def _set_default_error_reporting_mode():
    if _is_notebook():
        get_ipython().run_line_magic("xmode", "Context")
    else:
        sys.tracebacklimit = None


def turn_on_monitor(permanent: bool = False) -> None:
    """
    Run this to turn on the SlothPy Monitor utility.
    Set permamnet = True to save the settings permanently in the ini file.
    """
    settings.monitor = True
    if permanent:
        settings.save_settings()



def turn_off_monitor(permanent: bool = False) -> None:
    """
    Run this to turn off the SlothPy Monitor utility.
    Set permamnet = True to save the settings permanently in the ini file.
    """
    settings.monitor = False
    if permanent:
        settings.save_settings()


def set_plain_error_reporting_mode(permanent: bool = False) -> None:
    """
    Run this to set the custom SlothPy-style error printing without tracebacks.
    Set permamnet = True to save the settings permanently in the ini file.
    """

    settings.traceback = False
    if permanent:
        settings.save_settings()

def set_default_error_reporting_mode(permanent: bool = False) -> None:
    """
    Run this to set the default full error tracebacks.
    Set permamnet = True to save the settings permanently in the ini file.
    """

    settings.traceback = True
    if permanent:
        settings.save_settings()


def set_double_precision(permanent: bool = False) -> None:
    """
    Run this to set the double precision (float64, complex128) in computations.
    Set permamnet = True to save the settings permanently in the ini file.
    """

    settings.precision = "double"
    if permanent:
        settings.save_settings()


def set_single_precision(permanent: bool = False) -> None:
    """
    Run this to set the single precision (float64, complex128) in computations.
    Set permamnet = True to save the settings permanently in the ini file.
    """

    settings.precision = "single"
    if permanent:
        settings.save_settings()


def set_log_level(value: int = 0, permanent: bool = False) -> None:
    """
    Run this to set the logging level (float64, complex128) in SlotphPy.
    Set permamnet = True to save the settings permanently in the ini file.
    """

    settings.log_level = value
    if permanent:
        settings.save_settings()
