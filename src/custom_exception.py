import traceback
import sys

class CustomException(Exception):
    """
    A custom exception class that enriches standard exceptions with file name and line number details.

    Attributes:
        error_message (str): The formatted error message with contextual information.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes CustomException with a detailed error message.

        Args:
            error_message (str): The original error message.
            error_detail (module): Typically the `sys` module, used to extract traceback info.

        """
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        """
        Constructs a detailed error message including file name and line number.

        Args:
            error_message (str): The original error message.
            error_detail (module): Typically the `sys` module containing exception context.

        Returns:
            str: A formatted string with file name, line number, and the error message.
        """
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error in {file_name}, line {line_number}: {error_message}"

    def __str__(self):
        """
        Returns the string representation of the exception.

        Returns:
            str: The detailed error message.
        """
        return self.error_message
