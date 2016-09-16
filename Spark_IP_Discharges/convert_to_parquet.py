"""
### CODE OWNERS: Shea Parkes

### OBJECTIVE:
  Convert existing sas7bdat files into parquet files.

### DEVELOPER NOTES:
  Intended for demo purposes.
"""
import logging

import prm.meta.project
from prm.spark.app import SparkApp

LOGGER = logging.getLogger(__name__)

# =============================================================================
# LIBRARIES, LOCATIONS, LITERALS, ETC. GO ABOVE HERE
# =============================================================================



def main() -> int:
    """A function to enclose the execution of business logic."""
    LOGGER.info('About to do something awesome.')
    sparkapp = SparkApp('FL_IP_Demo')

    ### ADD NEW CODE HERE ###

    return 0


if __name__ == '__main__':
    # pylint: disable=wrong-import-position, wrong-import-order, ungrouped-imports
    import sys
    import prm.utils.logging_ext
    import prm.spark.defaults_prm

    prm.utils.logging_ext.setup_logging_stdout_handler()

    with SparkApp('FL_IP_Demo'):
        RETURN_CODE = main()

    sys.exit(RETURN_CODE)
