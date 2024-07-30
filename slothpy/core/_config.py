# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from os import cpu_count
from importlib import resources, resources as pkg_resources
from configparser import ConfigParser
from importlib import resources
from typing import Literal
from IPython import get_ipython
from numpy import int32, float32, complex64, int64, float64, complex128
from slothpy.core._slothpy_exceptions import SltInputError
from slothpy._general_utilities._system import _is_notebook
from slothpy._general_utilities._constants import RED, YELLOW, BLUE, GREEN, RESET


class SltSettings:

    def __init__(self):
        self._settings = {
            "number_cpu": 0,
            "number_threads": 1,
            "precision": "double",
            "monitor": False,
            "traceback": False,
            "log_level": 0,
        }
        config = ConfigParser()
        with resources.open_text("slothpy", "settings.ini") as file:
            config.read_file(file)

        # Update integer setting
        for key in ["number_cpu", "number_threads", "log_level"]:
            if key in config['DEFAULT']:
                self._settings[key] = config['DEFAULT'].getint(key)

        # Update boolean settings
        for key in ["monitor", "traceback"]:
            if key in config['DEFAULT']:
                self._settings[key] = config['DEFAULT'].getboolean(key)
        
        # Update string setting
        if "precision" in config['DEFAULT']:
            self._settings["precision"] = config['DEFAULT'].get("precision")

    def __repr__(self):
        return f"<{RED}SltSettings{RESET} object.>"

    def __str__(self):
        representation = f"{BLUE}Sloth{YELLOW}Py{RESET} Settings:\n"
        for name, value in self._settings.items():
            representation += f"{GREEN}{name}{RESET}: {value}\n"
        return representation.strip()
    
    @property
    def number_cpu(self):
        return self._settings["number_cpu"]

    @number_cpu.setter
    def number_cpu(self, value):
        if isinstance(value, int) and value >= 0 and value <= int(cpu_count()):
            self._settings["number_cpu"] = value
        else:
            raise SltInputError(
                ValueError(f"The number of CPUs has to be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
            )
        
    @property
    def number_threads(self):
        return self._settings["number_threads"]

    @number_threads.setter
    def number_threads(self, value):
        if isinstance(value, int) and value >= 0 and value <= int(cpu_count()):
            self._settings["number_threads"] = value
        else:
            raise SltInputError(
                ValueError(f"The number of Threads has to be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
            )

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
        
    @property
    def int(self):
        if self._settings["precision"] == "single":
            return int32
        elif self._settings["precision"] == "double":
            return int64
    
    @property
    def float(self):
        if self._settings["precision"] == "single":
            return float32
        elif self._settings["precision"] == "double":
            return float64
        
    @property
    def complex(self):
        if self._settings["precision"] == "single":
            return complex64
        elif self._settings["precision"] == "double":
            return complex128
        
    @property
    def numba_int(self):
        if self._settings["precision"] == "single":
            return "int32"
        elif self._settings["precision"] == "double":
            return "int64"
    
    @property
    def numba_float(self):
        if self._settings["precision"] == "single":
            return "float32"
        elif self._settings["precision"] == "double":
            return "float64"
        
    @property
    def numba_complex(self):
        if self._settings["precision"] == "single":
            return "complex64"
        elif self._settings["precision"] == "double":
            return "complex128"

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


# Settings init
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


def set_number_cpu(value: int = 0, permanent: bool = False) -> None:
    """
    Run this to set the number of CPUs used by SlotphPy caluclations.
    Set permanent = True to save the settings permanently in the .ini file.
    """

    settings.number_cpu = value
    if permanent:
        settings.save_settings()


def set_number_threads(value: int = 1, permanent: bool = False) -> None:
    """
    Run this to set the number of Threads in linear algebra libraries used by
    SlotphPy caluclations. Set permanent = True to save the settings
    permanently in the .ini file.
    """

    settings.number_threads = value
    if permanent:
        settings.save_settings()


def turn_on_monitor(permanent: bool = False) -> None:
    """
    Run this to turn on the SlothPy Monitor utility.
    Set permanent = True to save the settings permanently in the .ini file.
    """
    settings.monitor = True
    if permanent:
        settings.save_settings()


def turn_off_monitor(permanent: bool = False) -> None:
    """
    Run this to turn off the SlothPy Monitor utility.
    Set permanent = True to save the settings permanently in the .ini file.
    """
    settings.monitor = False
    if permanent:
        settings.save_settings()


def set_plain_error_reporting_mode(permanent: bool = False) -> None:
    """
    Run this to set the custom SlothPy-style error printing without tracebacks.
    Set permanent = True to save the settings permanently in the .ini file.
    """

    settings.traceback = False
    if permanent:
        settings.save_settings()


def set_default_error_reporting_mode(permanent: bool = False) -> None:
    """
    Run this to set the default full error tracebacks.
    Set permanent = True to save the settings permanently in the .ini file.
    """

    settings.traceback = True
    if permanent:
        settings.save_settings()


def set_double_precision(permanent: bool = False) -> None:
    """
    Run this to set the double precision (float64, complex128) in computations.
    Set permanent = True to save the settings permanently in the .ini file.
    """

    settings.precision = "double"
    if permanent:
        settings.save_settings()


def set_single_precision(permanent: bool = False) -> None:
    """
    Run this to set the single precision (float64, complex128) in computations.
    Set permanent = True to save the settings permanently in the .ini file.
    """

    settings.precision = "single"
    if permanent:
        settings.save_settings()


def set_log_level(value: int = 0, permanent: bool = False) -> None:
    """
    Run this to set the logging level (float64, complex128) in SlotphPy.
    Set permanent = True to save the settings permanently in the .ini file.
    """

    settings.log_level = value
    if permanent:
        settings.save_settings()
