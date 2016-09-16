"""
### CODE OWNERS: Shea Parkes

### OBJECTIVE:
  Demonstrate the DataFrame APIs

### DEVELOPER NOTES:
  Intended for demo purposes.
"""
import logging
from pathlib import Path

import pyspark.sql.functions as F
# pylint: disable=no-member

from prm.spark.app import SparkApp
import prm.spark.io_sas

LOGGER = logging.getLogger(__name__)

# =============================================================================
# LIBRARIES, LOCATIONS, LITERALS, ETC. GO ABOVE HERE
# =============================================================================



def main() -> int:
    """Demonstrate Spark APIs"""

    sparkapp = SparkApp('FL_IP_Demo')
    path_data = Path(
        r'k:/PRM/Reference_Data/xx-State_Discharge_Playground/'
    )


    ## "Load" up the discharges table
    df_discharges = sparkapp.load_df(
        path_data / 'fl_processedip_2014b.parquet'
    )
    print(df_discharges.count())


    ## Show the Domain Specific Language (DSL) API
    df_discharges.groupBy(
        'lob'
    ).agg(
        F.count('*').alias('row_cnt'),
        F.round(
            F.avg('los'),
            2,
        ).alias('avg_los'),
    ).orderBy(
        F.desc('row_cnt')
    ).show()


    ## Show the raw SQL API
    df_discharges.createOrReplaceTempView('discharges')
    sparkapp.session.sql('''
        SELECT
            lob,
            count(*) as row_cnt,
            round(avg(los), 2) as avg_los
        FROM discharges
        GROUP BY lob
        ORDER BY row_cnt desc
    ''').show()


    ## Show a lazy composition of the DSL API
    _df_grouped = df_discharges.groupBy(
        'lob'
    )
    _avg_los_expressions = [
        F.count('*').alias('row_cnt'),
        F.round(
            F.avg('los'),
            2,
        ).alias('avg_los'),
    ]
    _df_agged = _df_grouped.agg(
        *_avg_los_expressions
    )
    _df_ordered = _df_agged.orderBy(
        F.desc('row_cnt')
    )
    _df_ordered.show()


    ## Show the query plan
    _df_ordered.explain()


    ## Load up providers
    df_providers = sparkapp.load_df(
        path_data / 'fl_provider_2014.parquet'
    )
    print(df_providers.count())


    ## Show a join and aggregate
    sparkapp.spark_sql_shuffle_partitions = 12
    df_discharges.join(
        df_providers,
        on='providerid',
        how='left_outer',
    ).groupBy(
        df_providers.providername,
    ).agg(
        *_avg_los_expressions
    ).orderBy(
        F.desc('row_cnt')
    ).show()


    return 0


if __name__ == '__main__':
    # pylint: disable=wrong-import-position, wrong-import-order, ungrouped-imports
    import sys
    import prm.utils.logging_ext

    prm.utils.logging_ext.setup_logging_stdout_handler()

    with SparkApp('FL_IP_Demo'):
        RETURN_CODE = main()

    sys.exit(RETURN_CODE)
