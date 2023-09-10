"""Module for storing custom exception clasess with error reporting modes."""

from slothpy.general_utilities._constants import RED, GREEN, YELLOW, RESET
from slothpy.general_utilities.system import set_plain_error_reporting_mode

# The default is chosen to omit the tracebacks  completely. To change it use
# slt.set_default_error_reporting_mode method for the printing of tracebacks.
set_plain_error_reporting_mode()


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
    A custom exception class for runtime errors in computations.

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
        self.slt_message = (
            RED
            + "\nSlothComputationError"
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
        self.slt_message = (
            RED
            + "\nSlothSaveError"
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
        self.slt_message = (
            RED
            + "\nSlothReadError"
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
        self.slt_message = (
            RED
            + "\nSlothPlotError"
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
