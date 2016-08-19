"""
### CODE OWNERS: Ben Copeland, Brandon Patterson, Kyle Baird

### OBJECTIVE:
  Gather all of our features and then fit a number of models specified
  by the parameters provided

### DEVELOPER NOTES:
  <none>
"""

from math import sqrt
import logging
import itertools
from collections import namedtuple
import numpy as np
import pandas as pd

import pyspark
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import Window
from pyspark.sql.functions import lit, row_number, when, isnull, desc, col, sum  # pylint: disable=no-name-in-module
from pyspark.sql.types import FloatType

import prm.meta.project
import prm.spark.io_sas
from prm.spark.app import SparkApp

LOGGER = logging.getLogger(__name__)
PRM_META = prm.meta.project.parse_project_metadata()

PATH_COND_PRED_FEAT = PRM_META[160, "out"] / 'cond_cred_pred_feat.sas7bdat'
PATH_MEMBER_PARQUET = PRM_META[35, "out"] / 'member'
PATH_COND_TUNE_FEAT = PRM_META[160, "out"] / 'cond_cred_tune_feat.sas7bdat'
PATH_COND_TUNE_RESP = PRM_META[160, "out"] / 'cond_cred_tune_resp.sas7bdat'
PATH_DEMOG_FEATURES = PRM_META[160, "out"] / 'df_demog_long'
DIR_EXPORT = PRM_META[160, "out"]
DIR_CHECKPOINT = PRM_META[160, "temp"]

DEMOG_SCALAR = 2.0

RANK_LIST = [50]
ALPHA_LIST = [100.0]
LAMBDA_LIST = [7.0]

ModelParams = namedtuple("ModelParams", ["alpha", "lambda_", "rank"])

# =============================================================================
# LIBRARIES, LOCATIONS, LITERALS, ETC. GO ABOVE HERE
# =============================================================================



def main() -> int:
    """
        Reads in and formats the user/feature information,
        then fits a recommendation model
            for every combination of entries from RANK_LIST, ALPHA_LIST, and LAMBDA_LIST
    """

    ####################
    ### LAUNCH SPARK ###
    ####################

    sparkapp = SparkApp(PRM_META['pipeline_signature'])

    LOGGER.info('Importing tuning feature conditions table from SAS')
    df_tune_feat = prm.spark.io_sas.import_sas7bdat(
        sparkapp,
        PATH_COND_TUNE_FEAT,
        persist_level=pyspark.StorageLevel.MEMORY_AND_DISK_SER,
    )

    LOGGER.info('Importing tuning response conditions table from SAS')
    df_tune_resp = prm.spark.io_sas.import_sas7bdat(
        sparkapp,
        PATH_COND_TUNE_RESP,
    )

    LOGGER.info('Importing member table from public parquet')
    df_member = sparkapp.load_df(PATH_MEMBER_PARQUET).select(['member_id', 'gender'])

    LOGGER.info('Identifying gender-specific conditions')
    df_tune_feat_gender = df_tune_feat.join(
        df_member,
        "member_id"
        ).withColumn(
            'count_males',
            when(df_member.gender == 'M', 1).otherwise(0),
            ).withColumn(
                'count_females',
                when(df_member.gender == 'F', 1).otherwise(0)
                )

    df_gender_conds = df_tune_feat_gender.groupBy(
        'condition_name'
        ).mean(
            'count_males',
            'count_females',
            )

    df_gender_label = df_gender_conds.withColumn(
        'gender_specific',
        when(df_gender_conds['avg(count_males)'] >= .95, 'Male').otherwise(
            when(df_gender_conds['avg(count_females)'] >= .95, 'Female')
            .otherwise('Both')
            )
        ).cache()

    LOGGER.info('Limiting tuning response data to new conditons only')
    df_tune_resp_new_only = df_tune_resp.join(
        df_tune_feat,
        ['member_id', 'condition_name', 'chronic_flag'],
        how='left_outer',
        ).withColumn(
            'in_train',
            when(isnull(df_tune_feat['sqrt_adjusted_credibility']), 0).otherwise(1)
            ).filter('in_train = 0').drop(
                "log_adjusted_credibility"
                ).drop(
                    "sqrt_adjusted_credibility"
                    ).cache()

    LOGGER.info('Loading demographic features dataframe')
    df_demog = sparkapp.load_df(PATH_DEMOG_FEATURES)

    df_feature_all = munge_features(df_demog, df_tune_feat)

    ## Create user/condition mappings

    df_user_map = create_integer_mapping(df_feature_all, 'member_id', 'user_id')
    df_prod_map = create_integer_mapping(
        df_feature_all,
        ['feature_name', 'feature_value', 'chronic_flag'],
        'product_id',
        )

    df_ratings = create_ratings_df(df_feature_all, df_prod_map, df_user_map)

    df_tune_popular = df_tune_feat.select(
        'condition_name',
        'chronic_flag',
    ).groupBy(
        'condition_name',
        'chronic_flag',
    ).count()
    
    popular_window = Window.partitionBy('chronic_flag').orderBy(desc('count'))
    df_tune_popular_rank = df_tune_popular.withColumn(
        'pred_rank',
        row_number().over(popular_window)
    ).filter('pred_rank <= 30')
    
    df_member_popular = df_member.join(df_tune_popular_rank)
    df_member_popular_new_only= df_member_popular.join(
        df_tune_feat,
        ['member_id', 'condition_name', 'chronic_flag'],
        how='left_outer',
        ).withColumnRenamed(
            'sqrt_adjusted_credibility',
            'train_check',
            ).withColumn(
                'in_train',
                when(isnull('train_check'), 0).otherwise(1)
                ).filter('in_train = 0').drop(
                    "log_adjusted_credibility"
                    ).drop(
                        "sqrt_adjusted_credibility"
                        ).cache()
         
    member_popular_window = Window.partitionBy('member_id','chronic_flag').orderBy(desc('count'))           
    df_member_top_popular = df_member_popular_new_only.withColumn(
        'pred_rank',
        row_number().over(member_popular_window)
    ).filter('pred_rank <= 15')

    ## Predict iteratively over the parameter lists provided
    recommender = Recommender(sparkapp)
    recommender.set_inputs(df_user_map, df_prod_map, df_ratings)

    model_eval_results = {}
    for rank, alpha, lambda_ in itertools.product(RANK_LIST, ALPHA_LIST, LAMBDA_LIST):
        params = ModelParams(rank=rank, alpha=alpha, lambda_=lambda_)
        recommender.set_params(params)
        recommender.fit_model()
        model_eval = evaluate_predictions(
            recommender,
            df_tune_feat,
            df_member,
            df_gender_label,
            df_tune_resp_new_only,
			15,
            )
        model_eval_results[params] = model_eval[1]['avg(predicted)']

    best_iter_params = max(model_eval_results, key=model_eval_results.get)

    LOGGER.info('Importing prediction feature conditions table from SAS')
    df_pred_feat = prm.spark.io_sas.import_sas7bdat(
        sparkapp,
        PATH_COND_PRED_FEAT,
        persist_level=pyspark.StorageLevel.MEMORY_AND_DISK_SER,
    )
    df_pred_feat_all = munge_features(df_demog, df_pred_feat)

    ## Create user/condition mappings

    df_pred_feat_user_map = create_integer_mapping(df_pred_feat_all, 'member_id', 'user_id')
    df_pred_feat_prod_map = create_integer_mapping(
        df_pred_feat_all,
        ['feature_name', 'feature_value', 'chronic_flag'],
        'product_id',
        )

    df_pred_feat_ratings = create_ratings_df(
        df_pred_feat_all,
        df_pred_feat_prod_map,
        df_pred_feat_user_map
        )

    pred_recommender = Recommender(sparkapp)
    pred_recommender.set_inputs(df_pred_feat_user_map, df_pred_feat_prod_map, df_pred_feat_ratings)
    pred_recommender.set_params(best_iter_params)
    pred_recommender.fit_model()

    export_predictions(
        pred_recommender,
        df_pred_feat,
        df_member,
        df_gender_label
        )

    return 0


class Recommender:
    """Encapsulates the process/information needed to fit a recommendation model in Spark"""
    def __init__(self, sparkapp):
        self.sparkapp = sparkapp
        self.user_map = None
        self.prod_map = None

        # a bit of ugly duplication:
        # The df is needed for speed when explaining recommendations
        # The RDD is a necessary model input, which would be expensive to recalculate several times
        self.df_ratings = None
        self.rdd_ratings = None

        self.model = None
        self.rank = -1
        self.alpha = -1
        self.lambda_ = -1

    def set_params(self, params):
        """
        Set the parameters for the Recommender (wipes the current model)

        :param params: namedtuple containing some/all of 'rank', 'alpha', 'lambda'
        """
        if params.rank:
            self.rank = params.rank
        if params.alpha:
            self.alpha = params.alpha
        if params.lambda_:
            self.lambda_ = params.lambda_

        # reset the model because its parameters have changed
        self.model = None

    def set_inputs(self, df_user_map, df_product_map, df_ratings):
        """
        Set the input maps of users and products for later fitting (wipes the current model)

        :param
            df_user_map: a pyspark DataFrame mapping users to integers
            df_product_map: a pyspark DataFrame mapping products (i.e. conditions) to integers
            df_ratings: a pyspark DataFrame mapping
        """
        self.user_map = df_user_map
        self.prod_map = df_product_map
        self.df_ratings = df_ratings

        self.rdd_ratings = df_ratings.map(
            lambda row: Rating(int(row['user_id']), int(row['product_id']),
                               float(row['confidence'])))
        self.rdd_ratings.cache()
        self.sparkapp.context.setCheckpointDir(str(DIR_CHECKPOINT))
        self.rdd_ratings.checkpoint()

        # reset the model because inputs have changed
        self.model = None

    def fit_model(self):
        """Encapsulates a single fitting of the recommender engine based on stored inputs"""

        LOGGER.info(
            'Fitting recommender model with rank = %s, alpha = %s, lambda = %s',
            self.rank, self.alpha, self.lambda_
        )

        self.model = ALS.trainImplicit(
            self.rdd_ratings,
            rank=self.rank,
            iterations=10,
            lambda_=self.lambda_,
            alpha=self.alpha,
            blocks=int(sqrt(self.rdd_ratings.getNumPartitions())),
            nonnegative=True,
        )


    def explain_factors(self,user_id):
        """Explain the individual factors contributing to recommendations"""

        LOGGER.info('Extracting user/product factors')
        user_factors = self.model.userFeatures()
        pddf_user = user_factors.toDF(['user_id', 'factors']).toPandas() \
            .sort_values(by='user_id', ascending=True) \
            .set_index(['user_id'])
        product_factors = self.model.productFeatures()
        pddf_product = product_factors.toDF(['product_id', 'factors']).toPandas() \
            .sort_values(by='product_id', ascending=True) \
            .set_index(['product_id'])
        pddf_product_expand = pddf_product['factors'].apply(pd.Series)
        lambda_identity = np.identity(self.rank) * self.lambda_
        """Calculate prediction weight of each product-product pair"""
        user_input = self.df_ratings.filter(self.df_ratings.user_id == user_id).collect()
        confidence = np.identity(pddf_product.shape[0])

        for rows in user_input:
            confidence[rows[1]-1, rows[1]-1] = 1.0 + rows[2] * self.alpha

        # pylint: disable=invalid-name
        W = np.add(np.dot(np.dot(pddf_product_expand.transpose(), confidence),
                          pddf_product_expand), lambda_identity * len(user_input))
        W_inv = np.linalg.inv(W)
        # pylint: enable=invalid-name

        pairs = len(user_input) * self.prod_map.count() - len(user_input)

        df_sim_weight = pd.DataFrame(
            columns=('user', 'product_a', 'product_b', 'weight'), index=range(pairs))
        index = 0
        for i, j in itertools.permutations(range(self.prod_map.count()), 2):
            if confidence[j, j] > 1.0:
                sim_weight = float(
                    np.dot(
                        np.dot(pddf_product_expand[i:i + 1], W_inv),
                        pddf_product_expand[j:j + 1].transpose()
                    )
                ) * confidence[j, j]
                df_sim_weight.loc[index] = np.array([user_id, i, j, sim_weight])
                index += 1

        return df_sim_weight
            
    def explain_predictions(self):
        """Explain the individual factors contributing to recommendations"""

        LOGGER.info('Extracting user/product factors')
        user_factors = self.model.userFeatures()
        pddf_user = user_factors.toDF(['user_id', 'factors']).toPandas() \
            .sort_values(by='user_id', ascending=True) \
            .set_index(['user_id'])
        product_factors = self.model.productFeatures()
        pddf_product = product_factors.toDF(['product_id', 'factors']).toPandas() \
            .sort_values(by='product_id', ascending=True) \
            .set_index(['product_id'])
        pddf_product_expand = pddf_product['factors'].apply(pd.Series)
        lambda_identity = np.identity(self.rank) * self.lambda_

        LOGGER.info('Creating sample of prediction explanations')

        pddf_user_sample = pddf_user.sample(n=10)
        pddf_sim_weight_sample = pd.DataFrame(columns=('user', 'product_a', 'product_b', 'weight'))
        for users in pddf_user_sample.index:
            pddf_sim_weight_sample = pddf_sim_weight_sample.append(explain_factors(int(users)))
        df_sim_weight_sample = self.sparkapp.session.createDataFrame(pddf_sim_weight_sample)

        df_sim_weight_mapped_a = df_sim_weight_sample.join(
            self.user_map,
            df_sim_weight_sample.user == self.user_map.user_id
        ).join(
            self.prod_map,
            df_sim_weight_sample.product_a == self.prod_map.product_id
        ).select(
            self.user_map.member_id,
            self.prod_map.feature_name.alias('feature_name_a'),
            self.prod_map.feature_value.alias('feature_value_a'),
            df_sim_weight_sample.product_b,
            df_sim_weight_sample.weight,
        )

        df_sim_weight_mapped_b = df_sim_weight_mapped_a.join(
            self.prod_map,
            df_sim_weight_mapped_a.product_b == self.prod_map.product_id
        ).select(
            df_sim_weight_mapped_a.member_id,
            df_sim_weight_mapped_a.feature_name_a,
            df_sim_weight_mapped_a.feature_value_a,
            self.prod_map.feature_name.alias('feature_name_b'),
            self.prod_map.feature_value.alias('feature_value_b'),
            df_sim_weight_mapped_a.weight,
        )

        LOGGER.info('Exporting prediction explanations')
        prm.spark.io_sas.export_dataframe(
            df_sim_weight_mapped_b,
            DIR_EXPORT / '_'.join(('expl', str(int(self.rank)), str(int(self.alpha)),
                                   str(int(self.lambda_)) + '.sas7bdat'))
        )

def export_predictions(recommender, df_train, df_member, df_gender_label):
    """Export predictions for downstream use"""
    df_top_ratings = recommender.model.recommendProductsForUsers(30).flatMap(
        lambda row: row[1]).toDF()

    df_top_ratings_detail = df_top_ratings.join(
        recommender.user_map,
        df_top_ratings.user == recommender.user_map.user_id
        ).join(
            recommender.prod_map,
            df_top_ratings.product == recommender.prod_map.product_id
            ).select(
                recommender.user_map.member_id,
                recommender.prod_map.feature_name,
                recommender.prod_map.feature_value,
                recommender.prod_map.chronic_flag,
                df_top_ratings.rating,
                )

    df_top_ratings_new_only = df_top_ratings_detail.filter(
        df_top_ratings_detail['feature_name'] == 'condition_name'
    ).join(
        df_train.withColumnRenamed('condition_name', 'feature_value').drop('chronic_flag'),
        ['member_id', 'feature_value'],
        how='left_outer',
    ).withColumn(
        'in_train',
        when(isnull('sqrt_adjusted_credibility'), 0).otherwise(1)
    ).filter('in_train = 0')

    df_rating_gender = df_top_ratings_new_only.join(
        df_member,
        'member_id',
        how='left_outer',
    )

    df_rating_gender_specific = df_rating_gender.join(
        df_gender_label.select('condition_name', 'gender_specific'),
        df_rating_gender.feature_value == df_gender_label.condition_name,
        how='left_outer'
    ).drop('condition_name')

    df_preds_scrubbed = df_rating_gender_specific.filter(
        (df_rating_gender_specific.gender_specific == 'Both')
        | (
            (df_rating_gender_specific.gender_specific == 'Male')
            & (df_rating_gender_specific.gender == 'M')
            )
        | (
            (df_rating_gender_specific.gender_specific == 'Female')
            & (df_rating_gender_specific.gender == 'F')
            )
        )

    pred_window = Window.partitionBy('member_id').orderBy(desc('rating'))
    df_preds_rank = df_preds_scrubbed.withColumn(
        'pred_rank',
        row_number().over(pred_window)
    ).filter('pred_rank <= 10')

    LOGGER.info('Exporting ratings to SAS')
    prm.spark.io_sas.export_dataframe(
        df_preds_rank.select(
            'member_id',
            'feature_value',
            'feature_name',
            'chronic_flag',
            'rating',
            'pred_rank',
            ),
        DIR_EXPORT / 'preds.sas7bdat'
    )


def evaluate_predictions(recommender, df_train, df_member, df_gender_label, df_test_new_only, num_predictions):
    """Evaluate predictions on testing dataset"""
    df_top_ratings = recommender.model.recommendProductsForUsers(100).flatMap(
        lambda row: row[1]).toDF()

    df_top_ratings_detail = df_top_ratings.join(
        recommender.user_map,
        df_top_ratings.user == recommender.user_map.user_id
    ).join(
        recommender.prod_map,
        df_top_ratings.product == recommender.prod_map.product_id
    ).select(
        recommender.user_map.member_id,
        recommender.prod_map.feature_name,
        recommender.prod_map.feature_value,
        recommender.prod_map.chronic_flag,
        df_top_ratings.rating,
    )

    df_top_ratings_new_only = df_top_ratings_detail.filter(
        df_top_ratings_detail['feature_name'] == 'condition_name'
    ).join(
        df_train.withColumnRenamed('condition_name', 'feature_value').drop('chronic_flag'),
        ['member_id', 'feature_value'],
        how='left_outer',
    ).withColumn(
        'in_train',
        when(isnull('sqrt_adjusted_credibility'), 0).otherwise(1)
    ).filter('in_train = 0')

    df_rating_gender = df_top_ratings_new_only.join(
        df_member,
        'member_id',
        how='left_outer',
    )

    df_rating_gender_specific = df_rating_gender.join(
        df_gender_label.select('condition_name', 'gender_specific'),
        df_rating_gender.feature_value == df_gender_label.condition_name,
        how='left_outer'
    ).drop('condition_name')

    df_preds_scrubbed = df_rating_gender_specific.filter(
        (df_rating_gender_specific.gender_specific == 'Both')
        | (
            (df_rating_gender_specific.gender_specific == 'Male')
            & (df_rating_gender_specific.gender == 'M')
            )
        | (
            (df_rating_gender_specific.gender_specific == 'Female')
            & (df_rating_gender_specific.gender == 'F')
            )
        )

    pred_window = Window.partitionBy('member_id', 'chronic_flag').orderBy(desc('rating'))
    df_preds_rank = df_preds_scrubbed.withColumn(
        'pred_rank',
        row_number().over(pred_window)
    ).filter('pred_rank <= ' + str(num_predictions))

    df_test_w_preds = df_test_new_only.join(
        df_preds_rank.withColumnRenamed('feature_value', 'condition_name'),
        ['member_id', 'condition_name', 'chronic_flag'],
        how='left_outer',
    ).withColumn(
        'predicted',
        when(isnull('pred_rank'), 0).otherwise(1)
    )
    
    df_test_w_popular = df_test_new_only.join(
        df_member_top_popular,
        ['member_id','condition_name','chronic_flag'],
        how='left_outer',
    ).withColumn(
        'predicted',
        when(isnull('pred_rank'), 0).otherwise(1)
    )
    
    count_preds = df_test_w_popular.groupBy('chronic_flag').count()
    
    cumul_pred_pred = Window.partitionBy('chronic_flag').orderBy('pred_rank').rowsBetween(-9999999,0)
    mean_predicted = df_test_w_preds.join(
        count_preds.withColumnRenamed('count','total_count'),
        'chronic_flag',
    ).select(
        'chronic_flag',
        'pred_rank',
        'predicted',
        'total_count',
    ).groupBy('chronic_flag', 'total_count','pred_rank').count().withColumn(
        'pct_pred',
        col('count')/col('total_count'),
        ).filter(col('pred_rank') >= 1).withColumn(
            'cumul_pred',
            sum(col('pct_pred')).over(cumul_pred_pred)
            ).withColumn(
                'pred_type',
                lit('recommender'),
                ).drop('total_count')
    
    mean_predicted_popular = df_test_w_popular.join(
        count_preds.withColumnRenamed('count','total_count'),
        'chronic_flag',
    ).select(
        'chronic_flag',
        'pred_rank',
        'predicted',
        'total_count',
    ).groupBy('chronic_flag', 'total_count','pred_rank').count().withColumn(
        'pct_pred',
        col('count')/col('total_count'),
        ).filter(col('pred_rank') >= 1).withColumn(
            'cumul_pred',
            sum(col('pct_pred')).over(cumul_pred_pred)
            ).withColumn(
                'pred_type',
                lit('popular'),
                ).drop('total_count')
                
    preds_stack = mean_predicted.unionAll(mean_predicted_popular).cache()
    sparkapp.save_df(
        preds_stack,
        DIR_EXPORT / 'model_evaluation'
        )
        
    results = preds_stack.collect()

    return mean_predicted.collect()


def create_ratings_df(df_features, df_prod_map, df_user_map):
    """Create a ratings DataFrame that can be used to construct recommender rating inputs"""
    LOGGER.info('Creating ratings dataframe')
    df_ratings = df_features.join(
        df_user_map,
        df_features.member_id == df_user_map.member_id
    ).join(
        df_prod_map,
        ["feature_name", "feature_value", "chronic_flag"]
        ).select(
        df_user_map.user_id,
        df_prod_map.product_id,
        df_features.confidence.cast(FloatType()),
    )
    df_ratings.cache()
    return df_ratings


def create_integer_mapping(df_features, var_list, map_varname):
    """Create an integer mapping for each unique variable value"""
    LOGGER.info(
        'Creating %s integer mappings',
        map_varname,
    )
    _window = Window.orderBy(var_list)
    mapped_vals = df_features.select(var_list).distinct().withColumn(
        map_varname,
        row_number().over(_window),
    )

    return mapped_vals


def munge_features(df_demog, df_conds):
    """
    Do the initial munging of demographic and condition features
    so that they can be fed into the recommender engine
    """

    LOGGER.info('Munging feature tables')
    df_demog_munge = df_demog.withColumn(
        'confidence',
        df_demog.confidence.cast(FloatType()) * DEMOG_SCALAR,
    )
    # unionAll does not respect column names, only column order, so match the anicipated
    # condition table
    df_demog_munge_reorder = df_demog_munge.withColumn(
        'chronic_flag',
        lit(''),
        ).select(
            'member_id',
            'feature_value',
            'confidence',
            'chronic_flag',
            'feature_name',
            )

    df_cond_munge = df_conds.select(
        df_conds.member_id,
        df_conds.condition_name.alias('feature_value'),
        df_conds.sqrt_adjusted_credibility.alias('confidence'),
        df_conds.chronic_flag,
    ).withColumn('feature_name', lit('condition_name'))
    df_feature_all = df_cond_munge.unionAll(df_demog_munge_reorder)
    return df_feature_all


if __name__ == '__main__':
    # pylint: disable=wrong-import-position, wrong-import-order, ungrouped-imports
    import sys
    import prm.utils.logging_ext
    import prm.spark.defaults_prm

    prm.utils.logging_ext.setup_logging_stdout_handler()
    SPARK_DEFAULTS_PRM = prm.spark.defaults_prm.get_spark_defaults(PRM_META)

    with SparkApp(PRM_META['pipeline_signature'], **SPARK_DEFAULTS_PRM):
        RETURN_CODE = main()

    sys.exit(RETURN_CODE)
