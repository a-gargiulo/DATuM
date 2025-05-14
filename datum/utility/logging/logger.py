"""Define an all-purpose logger class."""
import sys
import datetime
import traceback
from typing import Optional
from colorama import init, Fore, Style

init()


class Logger:
    """Multi-purpose logger for information, warnings, errors, and debugging."""

    LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    # This will hold the singleton instance
    _instance = None

    def __new__(cls, log_to_file: Optional[str] = None, verbose: bool = True):
        """Ensure that only one instance of Logger is created."""
        if cls._instance is None:
            # If instance doesn't exist, create it
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_to_file: Optional[str] = None, verbose: bool = True) -> None:
        """
        Initialize the Logger.

        :param log_to_file: Optional path to a file where logs will be saved.
        :param verbose: Whether to also print logs to the console.
        """
        if not hasattr(self, '_initialized'):
            self.log_to_file = log_to_file
            self.verbose = verbose
            self._initialized = True

    def _current_time(self) -> str:
        return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")

    def _format_message(self, level: str, message: str) -> str:
        time = (
            Fore.MAGENTA +
            Style.BRIGHT +
            self._current_time() +
            Style.RESET_ALL
        )
        if level == "INFO":
            level_str = Fore.BLUE + Style.BRIGHT + level + Style.RESET_ALL
        elif level == "ERROR":
            level_str = Fore.RED + Style.BRIGHT + level + Style.RESET_ALL
        else:
            level_str = level

        return f"[{time}] [{level_str}] {message}"

    def _write(self, formatted_message: str) -> None:
        if self.verbose:
            print(formatted_message, file=sys.stderr)
        if self.log_to_file:
            try:
                with open(self.log_to_file, "a") as f:
                    f.write(formatted_message + "\n")
            except Exception as e:
                fallback_msg = (
                        f"[{self._current_time()}] [LOGGER ERROR] "
                        f"Could not write to log file: {e}"
                )
                print(fallback_msg, file=sys.stderr)

    def log(self, level: str, message: str) -> None:
        """Log a message with a custom level.

        :param level: Log level.
        :param message: Message to be logged.

        :raises ValueError: If the log level is invalid.
        """
        level = level.upper()
        if level not in self.LEVELS:
            raise ValueError(f"Invalid log level: {level}")
        formatted = self._format_message(level, message)
        self._write(formatted)

    def debug(self, message: str) -> None:
        """Log a debug-level message.

        :param message: Message to be logged.
        """
        self.log("DEBUG", message)

    def info(self, message: str) -> None:
        """Log an info-level message.

        :param message: Message to be logged.
        """
        self.log("INFO", message)

    def warning(self, message: str) -> None:
        """Log a warning-level message.

        :param message: Message to be logged.
        """
        self.log("WARNING", message)

    def error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log an error-level message, optionally with exception traceback.

        :param message: Message to be logged.
        :param exc: Exception traceback.
        """
        self.log("ERROR", message)
        if exc:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            for line in tb:
                self._write(line.rstrip())

    def critical(self, message: str) -> None:
        """Log a critical-level message.

        :param message: Message to be logged.
        """
        self.log("CRITICAL", message)
