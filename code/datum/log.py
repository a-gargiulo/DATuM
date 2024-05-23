"""This module provides functions for logging messages to the standard output
(terminal)."""
from typing import Callable, List


def print_title() -> None:
    """Print a fancy title."""
    title = (
        "------------------------------ Welcome to ------------------------------\n"
        "\n"
        "*******             **           **********               ****     ****\n"
        "/**////**           ****         /////**///               /**/**   **/**\n"
        "/**    /**         **//**            /**     **   **      /**//** ** /**\n"
        "/**    /**        **  //**           /**    /**  /**      /** //***  /**\n"
        "/**    /**       **********          /**    /**  /**      /**  //*   /**\n"
        "/**    **       /**//////**          /**    /**  /**      /**   /    /**\n"
        "/*******        /**     /**          /**    //******      /**        /**\n"
        "///////         //      //           //      //////       //         //\n"
        "\n"
        "------------------------------------------------------------------------\n"
        "Version 1.0.0\n"
        "\n"
        "© Aldo Gargiulo, 2023\n"
    )
    print(title)


def log_process(message: str, process_type: str) -> Callable:
    """A decorator to add the capability to a function to print a message to the
    console at the start and end of a process to track its status.

    :param message: A string containing the message to print.
    :param process_type: A string representing the hierarchy level of the decorated
        routine, i.e., whether it is a `main`, `sub`, or `subsub` routine.
    :return: The decorator function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_process_message(message, process_type)
            result = func(*args, **kwargs)
            end_process_message(f"{message} --> Done.")
            return result

        return wrapper

    return decorator


def create_ascii_table(headers: List[str], data: List[List[str]]) -> None:
    """Print an ASCII style table to nicely represent results.

    :param headers: A list of strings representing the header titles of the table.
    :param data: A list of lists of formatted strings. Each inner list in the outer
        list represents one row of the table.
    """
    # Determine the maximum width for each column
    column_widths = [max(len(str(item)) for item in col) for col in zip(headers, *data)]

    # Create the top border of the table
    table = ["-" * (width + 2) for width in column_widths]
    table = "+" + "+".join(table) + "+"

    # Add header row
    header_row = (
        "| "
        + " | ".join(
            header.center(width) for header, width in zip(headers, column_widths)
        )
        + " |"
    )
    table += (
        "\n"
        + header_row
        + "\n"
        + "+"
        + "+".join(["-" * (width + 2) for width in column_widths])
        + "+"
    )

    # Add data rows
    for row in data:
        data_row = (
            "| "
            + " | ".join(
                str(item).rjust(width) for item, width in zip(row, column_widths)
            )
            + " |"
        )
        table += (
            "\n"
            + data_row
            + "\n"
            + "+"
            + "+".join(["-" * (width + 2) for width in column_widths])
            + "+\n\n"
        )

    print(table)


def start_process_message(message: str, process_type: str = "main"):
    """Print a message at the start of a process.

    :param message: A string containing the message to print.
    :param process_type: A string representing the hierarchy level of the decorated
        routine, i.e., whether it is a `main`, `sub`, or `subsub` routine.
    """
    width = len(message) + 4
    symbol = ""
    if process_type == "main":
        symbol = "*"
    elif process_type == "sub":
        symbol = "/"
    elif process_type == "subsub":
        symbol = "~"

    border = symbol * width
    fancy_message = f"{border}\n{symbol} {message} {symbol}\n{border}\n\n"

    print(fancy_message)


def end_process_message(message: str):
    """Print a message at the end of a process.

    :param message: The message to print.
    """
    fancy_message = f"\t|\n\t| - - > {message}\n\n"
    print(fancy_message)
