"""Module for logging messages to the standard output."""
from typing import Callable, List


def print_title() -> None:
    """Print 'DATuM' title."""
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
        "© Aldo Gargiulo, 2023\n")
    print(title)


def log_process(msg: str, proc_type: str) -> Callable:
    """Function decorator for printing a message at the start and end of a function's
    execution.

    :param msg: The message to be printed.
    :param proc_type: The type of routine (`main`, `sub`, or `subsub`).

    :return: Function decorator.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            start_process_message(msg, proc_type)
            result = func(*args, **kwargs)
            end_process_message(f"{msg} --> Done.")
            return result

        return wrapper

    return decorator


def create_ascii_table(headers: List[str], data: List[List[str]]) -> None:
    """Print an ASCII style table of some data.

    :param headers: The header titles of the table.
    :param data: The data. The inner lists represent the rows of the table.
    """
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *data)]

    table = ["-" * (width + 2) for width in col_widths]
    table = "+" + "+".join(table) + "+"

    header_row = ("| " + " | ".join(
        header.center(width) for header, width in zip(headers, col_widths)) + " |")
    table += ("\n" + header_row + "\n" + "+" +
              "+".join(["-" * (width + 2) for width in col_widths]) + "+")

    for row in data:
        data_row = ("| " + " | ".join(
            str(item).rjust(width) for item, width in zip(row, col_widths)) + " |")
        table += ("\n" + data_row + "\n" + "+" +
                  "+".join(["-" * (width + 2) for width in col_widths]) + "+\n\n")

    print(table)


def start_process_message(msg: str, proc_type: str = "main"):
    """Print a message at the start of a routine.

    :param msg: The message to be printed.
    :param proc_type: The type of routine (`main`, `sub`, or `subsub`).
    """
    width = len(msg) + 4
    symbol = ""
    if proc_type == "main":
        symbol = "*"
    elif proc_type == "sub":
        symbol = "/"
    elif proc_type == "subsub":
        symbol = "~"

    border = symbol * width
    fancy_msg = f"{border}\n{symbol} {msg} {symbol}\n{border}\n\n"

    print(fancy_msg)


def end_process_message(msg: str):
    """Print a message at the end of a routine.

    :param msg: The message to be printed.
    """
    fancy_msg = f"\t|\n\t| - - > {msg}\n\n"
    print(fancy_msg)
