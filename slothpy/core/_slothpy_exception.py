"""Module for storing custom exception clasess."""


class SltError(Exception):
    """
    A custom exception class for errors connected to operations on .slt files.

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


class CompError(Exception):
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
