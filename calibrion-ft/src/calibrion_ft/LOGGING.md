# Logging Configuration

The `calibrion_ft` package includes a centralized logging configuration that provides consistent logging behavior across all modules.

## Usage

### Quick Start

```python
# Option 1: Import and use get_logger (recommended for most cases)
from calibrion_ft import get_logger

logger = get_logger(__name__)
logger.info("This is an info message")
logger.error("This is an error message")
```

```python
# Option 2: Use setup_logger for custom configuration
from calibrion_ft import setup_logger
import logging

logger = setup_logger(__name__, log_level=logging.DEBUG)
logger.debug("This is a debug message")
```

### Available Functions

#### `get_logger(name=None)`
- Returns a logger instance with default configuration
- If the logger doesn't exist, creates one with default settings
- Automatically uses the calling module's name if no name provided

#### `setup_logger(name=None, log_level=logging.INFO, format_string=None, include_timestamp=True)`
- Creates a new logger with custom configuration
- **Parameters:**
  - `name`: Logger name (auto-detected if None)
  - `log_level`: Logging level (e.g., `logging.INFO`, `logging.DEBUG`)
  - `format_string`: Custom format string for log messages
  - `include_timestamp`: Whether to include timestamp in logs

#### `configure_package_logging(level=logging.INFO)`
- Configures logging for the entire package
- Called automatically when the logging module is imported

### Log Levels

- `logging.DEBUG`: Detailed information for debugging
- `logging.INFO`: General information about program execution
- `logging.WARNING`: Something unexpected happened or a problem might occur
- `logging.ERROR`: A serious problem occurred
- `logging.CRITICAL`: A very serious error occurred

### Examples

#### Basic Logging
```python
from calibrion_ft import get_logger

logger = get_logger(__name__)

def process_data(data):
    logger.info(f"Processing {len(data)} items")
    
    for item in data:
        try:
            # Process item
            result = complex_operation(item)
            logger.debug(f"Processed item {item['id']}: {result}")
        except Exception as e:
            logger.error(f"Failed to process item {item['id']}: {e}")
            
    logger.info("Data processing completed")
```

#### Custom Logging Configuration
```python
from calibrion_ft import setup_logger
import logging

# Create a logger with custom format and debug level
logger = setup_logger(
    name="my_module",
    log_level=logging.DEBUG,
    format_string="[%(levelname)s] %(name)s: %(message)s",
    include_timestamp=False
)

logger.debug("This will show because level is DEBUG")
```

### Best Practices

1. **Use `get_logger(__name__)`** for most cases - it's simple and consistent
2. **Use appropriate log levels** - INFO for general flow, DEBUG for detailed debugging, ERROR for exceptions
3. **Include context in log messages** - add relevant IDs, counts, or state information
4. **Use f-strings for dynamic messages** - `logger.info(f"Processing {count} items")`
5. **Log exceptions with context** - `logger.exception(f"Failed to process {item_id}: {str(e)}")`

### Migration from Old Logging

If you have existing code using the old `log.logging_config` import:

```python
# OLD (broken import)
from log.logging_config import setup_logger

# NEW (working import)
from calibrion_ft.logging_config import setup_logger
# OR
from calibrion_ft import setup_logger
```

The function signatures remain the same, so existing logger setup calls should work without modification.