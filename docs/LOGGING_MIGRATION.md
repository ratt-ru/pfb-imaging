# Logging Migration Guide: From pyscilog to Rich-formatted Standard Logging

This guide explains how to migrate from pyscilog to the new Rich-formatted standard Python logging system in pfb-imaging.

## Overview

The new logging system provides:
- **Rich formatting** for beautiful console output
- **Full pyscilog compatibility** with the same API
- **Enhanced error handling** with rich tracebacks
- **Better performance** using standard Python logging
- **Graceful fallback** if Rich is not available

## Migration Steps

### 1. Update Dependencies

Rich has been added to `pyproject.toml`:
```toml
rich = ">=13.0.0"
```

### 2. Replace pyscilog imports

**Old (pyscilog):**
```python
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('COMPONENT_NAME')
```

**New (pfb.utils.logging):**
```python
from pfb.utils import logging as pfb_logging
pfb_logging.init('pfb')
log = pfb_logging.get_logger('COMPONENT_NAME')
```

### 3. Update logging calls

The logging interface remains **exactly the same**:

```python
# These work identically to pyscilog
log.info("Information message")
log.debug("Debug message")
log.warning("Warning message")
log.error("Error message")
log.error_and_raise("Error message", ValueError)  # Special pyscilog feature
```

### 4. File logging remains the same

```python
# Still works exactly like pyscilog
timestamp = time.strftime("%Y%m%d-%H%M%S")
logname = f'{opts.log_directory}/COMPONENT_{timestamp}.log'
pfb_logging.log_to_file(logname)
```

## API Compatibility

### Core Functions

| pyscilog | pfb.utils.logging | Notes |
|----------|------------------|-------|
| `pyscilog.init('pfb')` | `pfb_logging.init('pfb')` | Same interface |
| `pyscilog.get_logger('NAME')` | `pfb_logging.get_logger('NAME')` | Same interface |
| `pyscilog.log_to_file(filename)` | `pfb_logging.log_to_file(filename)` | Same interface |

### Logger Methods

| pyscilog | pfb.utils.logging | Notes |
|----------|------------------|-------|
| `log.info(msg)` | `log.info(msg)` | Same interface |
| `log.debug(msg)` | `log.debug(msg)` | Same interface |
| `log.warning(msg)` | `log.warning(msg)` | Same interface |
| `log.error(msg)` | `log.error(msg)` | Same interface |
| `log.error_and_raise(msg, exc)` | `log.error_and_raise(msg, exc)` | **Key compatibility feature** |

## New Features

### 1. Rich Formatting

The new system provides:
- **Syntax highlighting** in console output
- **Better tracebacks** with syntax highlighting and local variables
- **Timestamps** and **log levels** clearly displayed
- **Component names** color-coded
- **Automatic fallback** to standard logging if Rich unavailable

### 2. Enhanced Utilities

```python
# Log options dictionary (common pattern in pfb-imaging)
pfb_logging.log_options_dict(log, options, "Configuration Options")

# Create timestamped log files
log_file = pfb_logging.create_timestamped_log_file(log_dir, 'COMPONENT')

# Function call logging decorator
@pfb_logging.log_function_call(log)
def my_function(x, y):
    return x + y

# Context manager for temporary logging
with pfb_logging.TemporaryLogFile(log_dir, 'TEMP') as temp_log:
    temp_log.info("Temporary logging session")
```

### 3. Better Error Handling

```python
# Rich tracebacks with syntax highlighting and local variables
try:
    risky_operation()
except Exception as e:
    log.exception("Operation failed")  # Shows rich traceback
    # or
    log.error_and_raise("Operation failed", RuntimeError)
```

## Example Migration

### Before (pyscilog)
```python
import pyscilog
import time

# Initialize
pyscilog.init('pfb')
log = pyscilog.get_logger('INIT')

# Set up file logging
timestamp = time.strftime("%Y%m%d-%H%M%S")
logname = f'{opts.log_directory}/INIT_{timestamp}.log'
pyscilog.log_to_file(logname)

# Log options
log.info('Input Options:')
for key in opts.keys():
    log.info('     %25s = %s' % (key, opts[key]))

# Error handling
try:
    assert condition
except Exception as e:
    log.error_and_raise("Assertion failed", AssertionError)
```

### After (pfb.utils.logging)
```python
from pfb.utils import logging as pfb_logging
import time

# Initialize
pfb_logging.init('pfb')
log = pfb_logging.get_logger('INIT')

# Set up file logging
timestamp = time.strftime("%Y%m%d-%H%M%S")
logname = f'{opts.log_directory}/INIT_{timestamp}.log'
pfb_logging.log_to_file(logname)

# Log options (enhanced utility)
pfb_logging.log_options_dict(log, opts, 'Input Options')

# Error handling (same interface, better tracebacks)
try:
    assert condition
except Exception as e:
    log.error_and_raise("Assertion failed", AssertionError)
```

## Benefits of Migration

1. **Better Console Output**: Rich formatting makes logs more readable
2. **Enhanced Debugging**: Rich tracebacks show local variables and syntax highlighting
3. **Performance**: Standard Python logging is faster than pyscilog
4. **Maintainability**: No external dependency on pyscilog Git repository
5. **Compatibility**: Drop-in replacement with same API
6. **Future-proof**: Based on standard Python logging infrastructure

## Testing the Migration

Run the example script to test the new logging system:

```bash
python examples/logging_example.py
```

This will demonstrate:
- Console output with Rich formatting
- File logging compatibility
- Error handling with `error_and_raise`
- New utility functions
- Context manager usage

## Troubleshooting

### If Rich is not available
The system automatically falls back to standard Python logging if Rich is not installed. All functionality remains available, just without the enhanced formatting.

### File logging issues
The new system creates log directories automatically and handles file permissions the same way as pyscilog.

### Performance considerations
The new system is generally faster than pyscilog, especially for high-volume logging scenarios.

## Migration Checklist

- [ ] Update `pyproject.toml` to include Rich dependency
- [ ] Replace `import pyscilog` with `from pfb.utils import logging as pfb_logging`
- [ ] Update `pyscilog.init()` calls to `pfb_logging.init()`
- [ ] Update `pyscilog.get_logger()` calls to `pfb_logging.get_logger()`
- [ ] Update `pyscilog.log_to_file()` calls to `pfb_logging.log_to_file()`
- [ ] Test with `examples/logging_example.py`
- [ ] Run existing tests to ensure compatibility
- [ ] Remove pyscilog dependency from `pyproject.toml` after migration is complete