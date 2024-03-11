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

"""Module for storing custom exception classes with error reporting modes."""
from functools import wraps
from typing import Literal
from slothpy._general_utilities._constants import RED, GREEN, YELLOW, RESET


class SltInputError(Exception):
    """
    A custom exception class for errors in input data.

    Parameters
    ----------
    None
    """

    def __init__(self, exception: Exception, message: str = ""):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """
        self.error_type = type(exception).__name__
        self.error_message = str(exception)
        self.slt_message = f"{RED}\nSlothInputError{RESET}, {YELLOW}{self.error_type}{RESET}: {self.error_message} "
        self.final_message = f"{self.slt_message} {message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """
        return self.final_message


class SltFileError(Exception):
    """
    A custom exception class for errors connected to operations on .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        file : str
            A file to which the error corresponds.
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """

        self.error_type = type(exception).__name__
        self.error_message = str(exception)
        self.slt_message = f"{RED}\nSlothFileError{RESET}, {GREEN}File{RESET} '{file}', {YELLOW}{self.error_type}{RESET}: {self.error_message} "
        self.final_message = f"{self.slt_message} {message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """

        return self.final_message


class SltCompError(Exception):
    """
    A custom exception class for runtime errors during computations.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        file : str
            A file to which the error corresponds.
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """
        self.error_type = type(exception).__name__
        self.error_message = str(exception)
        self.slt_message = f"{RED}\nSlothComputationError{RESET}, {GREEN}File{RESET} '{file}', {YELLOW}{self.error_type}{RESET}: {self.error_message} "
        self.final_message = f"{self.slt_message} {message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """
        return self.final_message


class SltSaveError(Exception):
    """
    A custom exception class for errors in saving data to .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        file : str
            A file to which the error corresponds.
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """
        self.error_type = type(exception).__name__
        self.error_message = str(exception)
        self.slt_message = f"{RED}\nSlothSaveError{RESET}, {GREEN}File{RESET} '{file}', {YELLOW}{self.error_type}{RESET}: {self.error_message}"
        self.final_message = f"{self.slt_message} {message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """
        return self.final_message


class SltReadError(Exception):
    """
    A custom exception class for errors in reading data from .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        file : str
            A file to which the error corresponds.
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """
        self.error_type = type(exception).__name__
        self.error_message = str(exception)
        self.slt_message = f"{RED}\nSlothReadError{RESET}, {GREEN}File{RESET} '{file}', {YELLOW}{self.error_type}{RESET}: {self.error_message} "
        self.final_message = f"{self.slt_message} {message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """
        return self.final_message
    

class SltPlotError(Exception):
    """
    A custom exception class for errors in data plotting from .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        file : str
            A file to which the error corresponds.
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """
        self.error_type = type(exception).__name__
        self.error_message = str(exception)
        self.slt_message = f"{RED}\nSlothPlotError{RESET}, {GREEN}File{RESET} '{file}', {YELLOW}{self.error_type}{RESET}: {self.error_message} "
        self.final_message = f"{self.slt_message} {message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """
        return self.final_message
    

def slothpy_exc(slt_exception: Literal["SltFileError", "SltCompError", "SltSaveError", "SltReadError", "SltInputError", "SltPlotError"], slt_message: str = "") -> callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                exception_mapping = {
                    "SltInputError": SltInputError,
                    "SltFileError": SltFileError,
                    "SltCompError": SltCompError,
                    "SltSaveError": SltSaveError,
                    "SltReadError": SltReadError,
                    "SltPlotError": SltPlotError,
                }
                if slt_exception in exception_mapping:
                    if slt_exception == "SltInputError":
                        raise exception_mapping[slt_exception](exc, slt_message) from None
                    else:
                        raise exception_mapping[slt_exception](args[0]._hdf5, exc, slt_message) from None
                else:
                    raise ValueError(f"Unsupported {RED}SltException{RESET} provided.") from None
        return wrapper
    return decorator

