"""
### CODE OWNERS: Shea Parkes

### OBJECTIVE:
  Convert existing sas7bdat files into parquet files.

### DEVELOPER NOTES:
  Intended for demo purposes.
"""
import logging
from pathlib import Path

from prm.spark.app import SparkApp
import prm.spark.io_sas

LOGGER = logging.getLogger(__name__)

# =============================================================================
# LIBRARIES, LOCATIONS, LITERALS, ETC. GO ABOVE HERE
# =============================================================================



def main() -> int:
    """A function to enclose the execution of business logic."""
    LOGGER.info('Converting existing sas7bdat files to parquet ~files')
    sparkapp = SparkApp('FL_IP_Demo')

    path_data = Path(r'k:/PRM/Reference_Data/xx-State_Discharge_Playground/')
    for path_sas7bdat in path_data.glob('*.sas7bdat'):
        prm.spark.io_sas.convert_sas7bdat_to_parquet(sparkapp, path_sas7bdat)

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
