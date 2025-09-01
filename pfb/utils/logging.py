"""
Logging utilities for PFB-Imaging using standard Python logging with Rich formatting.

This module provides a drop-in replacement for pyscilog with enhanced
formatting using the Rich library for better console output.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Type, Union
from functools import wraps

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.text import Text
    from rich.traceback import install as install_rich_traceback
    RICH_AVAILABLE = True
    # Install rich traceback handling
    install_rich_traceback(show_locals=False)
except ImportError:
    RICH_AVAILABLE = False


class PFBLogger:
    """
    Enhanced logger class that provides pyscilog-compatible interface
    with Rich formatting capabilities.
    """
    
    def __init__(self, name: str, app_name: str = 'pfb'):
        self.name = name
        self.app_name = app_name
        self.logger = logging.getLogger(f"{app_name}.{name}")
        # self._console = Console(width=120, force_terminal=True) if RICH_AVAILABLE else None
        self._console = Console(force_terminal=True) if RICH_AVAILABLE else None
        
    def info(self, message: str, *args, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)
        
    def warn(self, message: str, *args, **kwargs) -> None:
        """Alias for warning for compatibility."""
        self.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log a critical message."""
        self.logger.critical(message, *args, **kwargs)
        
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log an exception with traceback."""
        self.logger.exception(message, *args, **kwargs)
    
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
        self.logger.error(message, *args, **kwargs)
        raise exception_type(message)


class PFBLoggingManager:
    """
    Global logging manager that handles initialization and configuration.
    """
    
    def __init__(self):
        self._initialized = False
        self._app_name = 'pfb'
        self._loggers: Dict[str, PFBLogger] = {}
        self._log_files: Dict[str, str] = {}
        self._console = Console(width=120, force_terminal=True) if RICH_AVAILABLE else None
        self._root_logger = logging.getLogger(self._app_name)
        
    def init(self, app_name: str = 'pfb', log_level: Union[str, int] = logging.INFO) -> None:
        """
        Initialize the logging system.
        
        Args:
            app_name: Name of the application (default: 'pfb')
            log_level: Logging level (default: INFO)
        """
        if self._initialized:
            return
            
        self._app_name = app_name
        self._root_logger = logging.getLogger(app_name)
        
        # Set up the root logger
        self._root_logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self._root_logger.handlers[:]:
            self._root_logger.removeHandler(handler)
        
        # Create console handler with Rich formatting if available
        if RICH_AVAILABLE:
            console_handler = RichHandler(
                console=self._console,
                show_time=False,  # Disable Rich's built-in time formatting
                show_level=False,  # Disable Rich's built-in level formatting
                show_path=False,  # Disable path to save space for longer messages
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                markup=True
            )
            console_handler.setLevel(log_level)
            
            # Custom format for longer lines with prominent timestamp and aligned levels
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
        else:
            # Fallback to standard handler if Rich is not available
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
        
        self._root_logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate messages
        self._root_logger.propagate = False
        
        self._initialized = True
    
    def get_logger(self, component_name: str) -> PFBLogger:
        """
        Get a logger for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            PFBLogger instance
        """
        if not self._initialized:
            self.init()
            
        if component_name not in self._loggers:
            self._loggers[component_name] = PFBLogger(component_name, self._app_name)
            
        return self._loggers[component_name]
    
    def log_to_file(self, filename: Union[str, Path], component_name: Optional[str] = None) -> None:
        """
        Add file logging to the specified file.
        
        Args:
            filename: Path to the log file
            component_name: Optional component name to restrict file logging to specific component
        """
        if not self._initialized:
            self.init()
            
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
        if component_name:
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
            
        self._root_logger.setLevel(level)
        
        # Update all handlers
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
_logging_manager = PFBLoggingManager()

# Public API functions for pyscilog compatibility
def init(app_name: str = 'pfb', log_level: Union[str, int] = logging.INFO) -> None:
    """
    Initialize the logging system.
    
    Args:
        app_name: Name of the application (default: 'pfb')
        log_level: Logging level (default: INFO)
    """
    _logging_manager.init(app_name, log_level)


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
    logger.info(f'{title}:')
    for key, value in options.items():
        logger.info(f'     {key:>25s} = {value}')


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
        self.logger.info(f"Logging session ended")