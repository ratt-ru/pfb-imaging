"""
Logging utilities for PFB-Imaging using standard Python logging with Rich formatting.

This module provides a drop-in replacement for pyscilog with enhanced
formatting using the Rich library for better console output.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Type, Union
from functools import wraps

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.table import Table, Column
from rich.panel import Panel
from rich.text import Text

rich_console = Console()

install_rich_traceback(console=rich_console, show_locals=False)


class PFBLogger(logging.Logger):
    """
    Enhanced logger class that provides pyscilog-compatible interface
    with Rich formatting capabilities.
    """

    def error_and_raise(self, message: str, exception_type: Type[Exception] = Exception, *args, **kwargs) -> None:
        """
        Log an error message and raise an exception.

        This provides compatibility with pyscilog's error_and_raise method.

        Args:
            message: The error message to log and include in the exception
            exception_type: The type of exception to raise (default: Exception)
            *args: Additional arguments for the logger
            **kwargs: Additional keyword arguments for the logger
        """

        bold_message = f"[bold red]{message}[/bold red]"
        self.error(bold_message, *args, **kwargs)

        raise exception_type(message)

# Any logger created hereafter will be an instance of PFBLogger.
logging.setLoggerClass(PFBLogger)

class LoggingManager:
    """
    Global logging manager that handles initialization and configuration.
    """

    def __init__(self, app_name: str, log_level: Union[str, int] = logging.INFO) -> None:
        """
        Initialize the logging system.

        Args:
            app_name: Name of the application
            log_level: Logging level of the default console handler.
        """
        self._app_name = app_name
        self._loggers: Dict[str, PFBLogger] = {}
        self._log_files: Dict[str, str] = {}
        self._console = rich_console
        self._root_logger = logging.getLogger(self._app_name)

        # Set up the root logger - this should always be set to DEBUG level to ensure that all
        # log messages are dispatched to the relevant handlers.
        self._root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        for handler in self._root_logger.handlers:
            self._root_logger.removeHandler(handler)

        # Create console handler with Rich formatting. The RichHandler has limited configuration
        # options. If needed, it is possible to subclass it and modify its methods.
        console_handler = RichHandler(
            console=self._console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True
        )
        console_handler.setLevel(log_level)

        # NOTE(JSKenyon): Removed the formatter for now as the RichHandler defaults seem sensible.
        # formatter = logging.Formatter(
        #     fmt="%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        #     style="{"
        # )
        # console_handler.setFormatter(formatter)

        self._root_logger.addHandler(console_handler)

        # Prevent propagation from the application logger into the Python root logger i.e.
        # the application logger is the final authority.
        self._root_logger.propagate = False


    def get_logger(self, component_name: str) -> PFBLogger:
        """
        Get a logger for a specific component.

        Args:
            component_name: Name of the component

        Returns:
            PFBLogger instance
        """
        if component_name not in self._loggers:
            self._loggers[component_name] = logging.getLogger(name=f"{self._app_name}.{component_name}")

        return self._loggers[component_name]

    def log_to_file(self, filename: Union[str, Path], component_name: Optional[str] = None) -> None:
        """
        Add file logging to the specified file.

        Args:
            filename: Path to the log file
            component_name: Optional component name to restrict file logging to specific component
        """
        filename = Path(filename)

        # Create directory if it doesn't exist
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(filename, mode='a')
        file_handler.setLevel(logging.DEBUG)  # File gets all messages

        # File formatter (more detailed than console)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)-15s - %(levelname)-8s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Add to appropriate logger
        if component_name:  # Do we want to support this?
            # Add to specific component logger
            logger = self.get_logger(component_name)
            logger.logger.addHandler(file_handler)
        else:
            # Add to root logger (affects all components)
            self._root_logger.addHandler(file_handler)

        # Store the file reference
        key = component_name if component_name else 'root'
        self._log_files[key] = str(filename)

        # Log that we've started logging to file
        if component_name:
            logger = self.get_logger(component_name)
            logger.info(f"Logging to file: {filename}")
        else:
            # For root logger, log directly to avoid duplication
            # since child loggers will inherit the file handler
            self._root_logger.info(f"Logging to file: {filename}")

    def set_log_level(self, level: Union[str, int]) -> None:
        """
        Set the logging level for all loggers.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        self._root_logger.setLevel(level)  # Should the application logger ever change level?

        # Update all handlers which are not instances of logging.FileHandler.
        for handler in self._root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)

    def get_log_files(self) -> Dict[str, str]:
        """Get dictionary of active log files."""
        return self._log_files.copy()

    def close_log_files(self) -> None:
        """Close all file handlers."""
        for handler in self._root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self._root_logger.removeHandler(handler)

        # Clear file references
        self._log_files.clear()


# Global manager instance
_logging_manager = LoggingManager("pfb")


def get_logger(component_name: str) -> PFBLogger:
    """
    Get a logger for a specific component.

    Args:
        component_name: Name of the component

    Returns:
        PFBLogger instance
    """
    return _logging_manager.get_logger(component_name)


def log_to_file(filename: Union[str, Path], component_name: Optional[str] = None) -> None:
    """
    Add file logging to the specified file.

    Args:
        filename: Path to the log file
        component_name: Optional component name to restrict file logging to specific component
    """
    _logging_manager.log_to_file(filename, component_name)


def set_log_level(level: Union[str, int]) -> None:
    """
    Set the logging level for all loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    _logging_manager.set_log_level(level)


def get_log_files() -> Dict[str, str]:
    """Get dictionary of active log files."""
    return _logging_manager.get_log_files()


def close_log_files() -> None:
    """Close all file handlers."""
    _logging_manager.close_log_files()


# Utility functions for common logging patterns
def log_function_call(logger: PFBLogger):
    """
    Decorator to log function calls with arguments.

    Args:
        logger: PFBLogger instance to use for logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log function entry
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed with error: {e}")
                raise
        return wrapper
    return decorator


def log_options_dict(logger: PFBLogger, options: Dict[str, Any], title: str = "Options") -> None:
    """
    Log a dictionary of options in a formatted way.

    Args:
        logger: PFBLogger instance to use for logging
        options: Dictionary of options to log
        title: Title for the options section
    """

    name_col = Column(justify="left")
    spacer_col = Column(justify="center")
    val_col = Column(justify="left")

    options_table = Table.grid(name_col, spacer_col, val_col, expand=True)

    for key, value in options.items():
        options_table.add_row(f"[green]{key}[/green]", "|", f"[white]{value}[/white]")

    with rich_console.capture() as capture:
        rich_console.print(Panel(options_table, style="cyan", title="Inputs"))

    str_output = Text.from_ansi(capture.get())
    rich_console.print(str_output.markup)
    logger.debug(str_output)

def create_timestamped_log_file(log_directory: Union[str, Path], component_name: str) -> str:
    """
    Create a timestamped log file path compatible with pyscilog pattern.

    Args:
        log_directory: Directory where log files should be stored
        component_name: Name of the component for the log file

    Returns:
        Path to the log file
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    logname = log_directory / f'{component_name}_{timestamp}.log'
    return str(logname)


# Context manager for temporary log file
class TemporaryLogFile:
    """Context manager for temporary log file setup."""

    def __init__(self, log_directory: Union[str, Path], component_name: str):
        self.log_directory = log_directory
        self.component_name = component_name
        self.log_file = None
        self.logger = None

    def __enter__(self):
        self.log_file = create_timestamped_log_file(self.log_directory, self.component_name)
        self.logger = get_logger(self.component_name)
        log_to_file(self.log_file, self.component_name)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Error occurred: {exc_val}")
        self.logger.info("Logging session ended")
