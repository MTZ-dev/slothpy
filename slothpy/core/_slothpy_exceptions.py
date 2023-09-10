"""Module for storing custom exception clasess."""
import sys

# import IPython
from slothpy.core._constants import RED, GREEN, YELLOW, RESET

# Set a custom traceback limit for printing the SltErrors
# for system and Jupyter Notebook. Edit it for debugging.
sys.tracebacklimit = 0
# IPython.get_ipython().run_line_magic("xmode", "Plain")


class SltFileError(Exception):
    """
    A custom exception class for errors connected to operations on .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        """Initialize for the custom message printing.

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
        self.slt_message = (
            RED
            + "\nSlothFileError"
            + RESET
            + ", "
            + GREEN
            + "File "
            + RESET
            + f'"{file}", '
            + YELLOW
            + f"{self.error_type}"
            + RESET
            + f": {self.error_message} \n"
        )
        self.final_message = self.slt_message + message
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
        return str(self.final_message)


class SltCompError(Exception):
    """
    A custom exception class for runtime errors connected with computations.

    Parameters
    ----------
    None
    """

    def __init__(self, message: str):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        message : str
            A message to be printed.
        """
        super().__init__(message)


class SltPlotError(Exception):
    pass
