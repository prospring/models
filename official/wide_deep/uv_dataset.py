from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
from six.moves import urllib
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.utils.flags import core as flags_core

TRAINING_FILE = 'uv_sampled_train.csv'
EVAL_FILE = 'uv_sampled_test.csv'
_CSV_COLUMNS = ["brand_id", "cate_id", "cate_level1_id", "col_v_xd_sum",
                "crt_v_xd_sum",  "disc", "ds", "gmv_xd_sum",
                "item_id", "label", "odr_v_xd_sum", "price_avg", "price_daily",
                "pv_xd_sum", "sku_crt_15d", "sku_crt_diff", "sku_gmv_15d",
                "sku_gmv_diff", "sku_id", "sku_ord_15d", "sku_ord_diff",
                "sku_sls_15d", "sku_sls_diff", "sls_v_xd_sum", "user_cate_col",
                "user_cate_crt", "user_cate_pay", "user_cate_pv", "user_id",
                "user_brand_col", "user_brand_crt", "user_brand_pay", "user_brand_pv"
                ]

_CSV_COLUMN_TYPE = [tf.uint64, tf.uint64, tf.uint64, tf.uint64,
                    tf.uint64, tf.float64, tf.string, tf.float64,
                    tf.uint64, tf.uint64, tf.uint64, tf.float64, tf.float64,
                    tf.uint64, tf.uint64, tf.float64, tf.float64,
                    tf.float64, tf.uint64, tf.uint64, tf.float64,
                    tf.uint64, tf.float64, tf.uint64, tf.uint64,
                    tf.uint64, tf.uint64, tf.uint64, tf.uint64,
                    tf.uint64, tf.uint64, tf.uint64, tf.uint64]

_CSV_COLUMN_DEFAULTS = [[0]*5, [0.0], [''], [0.0]
                        [0]*3, [0.0]*2,
                        [0]*2, [0.0]*2,
                        [0.0], [0]*2, [0.0],
                        [0], [0.0], [0]*2,
                        [0]*8]







_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 7580357,
    'validation': 1532490,
}

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous variable columns

  sku_ord_15d = tf.feature_column.numeric_column('sku_ord_15d')
  sku_sls_15d = tf.feature_column.numeric_column('sku_sls_15d')
  sku_crt_15d = tf.feature_column.numeric_column('sku_crt_15d')
  odr_v_xd_sum = tf.feature_column.numeric_column('odr_v_xd_sum')
  sls_v_xd_sum = tf.feature_column.numeric_column('sls_v_xd_sum')
  crt_v_xd_sum = tf.feature_column.numeric_column('crt_v_xd_sum')
  pv_xd_sum = tf.feature_column.numeric_column('pv_xd_sum')
  col_v_xd_sum = tf.feature_column.numeric_column('col_v_xd_sum')
  user_cate_pv = tf.feature_column.numeric_column('user_cate_pv')
  user_cate_crt = tf.feature_column.numeric_column('user_cate_crt')
  user_cate_col = tf.feature_column.numeric_column('user_cate_col')
  user_cate_pay = tf.feature_column.numeric_column('user_cate_pay')
  user_brand_pv = tf.feature_column.numeric_column('user_brand_pv')
  user_brand_crt = tf.feature_column.numeric_column('user_brand_crt')
  user_brand_col = tf.feature_column.numeric_column('user_brand_col')
  user_brand_pay = tf.feature_column.numeric_column('user_brand_pay')
  price_avg = tf.feature_column.numeric_column('price_avg')
  price_daily = tf.feature_column.numeric_column('price_daily')
  disc = tf.feature_column.numeric_column('disc')
  sku_gmv_15d = tf.feature_column.numeric_column('sku_gmv_15d')
  sku_ord_diff = tf.feature_column.numeric_column('sku_ord_diff')
  sku_gmv_diff = tf.feature_column.numeric_column('sku_gmv_diff')
  sku_sls_diff = tf.feature_column.numeric_column('sku_sls_diff')
  sku_crt_diff = tf.feature_column.numeric_column('sku_crt_diff')
  gmv_xd_sum = tf.feature_column.numeric_column('gmv_xd_sum')


  # hashing features:
  disc_hash = tf.feature_column.categorical_column_with_hash_bucket('disc', hash_bucket_size=200)

  price_avg_hash = tf.feature_column.categorical_column_with_hash_bucket('price_avg', hash_bucket_size=_HASH_BUCKET_SIZE)

  price_daily_hash = tf.feature_column.categorical_column_with_hash_bucket('price_daily', hash_bucket_size=_HASH_BUCKET_SIZE)

  sku_crt_15d_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_crt_15d', hash_bucket_size=_HASH_BUCKET_SIZE)

  sku_crt_diff_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_crt_diff', hash_bucket_size=_HASH_BUCKET_SIZE)

  sku_gmv_15d_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_gmv_15d', hash_bucket_size=_HASH_BUCKET_SIZE)


  sku_gmv_diff_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_gmv_diff', hash_bucket_size=_HASH_BUCKET_SIZE)
  sku_ord_15d_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_ord_15d', hash_bucket_size=_HASH_BUCKET_SIZE)
  sku_ord_diff_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_ord_diff', hash_bucket_size=_HASH_BUCKET_SIZE)
  sku_sls_15d_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_sls_15d', hash_bucket_size=_HASH_BUCKET_SIZE)
  sku_sls_diff_hash = tf.feature_column.categorical_column_with_hash_bucket('sku_sls_diff', hash_bucket_size=_HASH_BUCKET_SIZE)
  user_cate_pv_hash = tf.feature_column.categorical_column_with_hash_bucket('user_cate_pv', hash_bucket_size=_HASH_BUCKET_SIZE)
  user_cate_pay_hash = tf.feature_column.categorical_column_with_hash_bucket('user_cate_pay', hash_bucket_size=_HASH_BUCKET_SIZE)

  user_brand_pv_hash = tf.feature_column.categorical_column_with_hash_bucket('user_brand_pv', hash_bucket_size=_HASH_BUCKET_SIZE)
  user_brand_pay_hash = tf.feature_column.categorical_column_with_hash_bucket('user_brand_pay', hash_bucket_size=_HASH_BUCKET_SIZE)
  gmv_xd_sum_hash = tf.feature_column.categorical_column_with_hash_bucket('gmv_xd_sum', hash_bucket_size=_HASH_BUCKET_SIZE)

  pv_xd_sum_hash = tf.feature_column.categorical_column_with_hash_bucket('pv_xd_sum', hash_bucket_size=100)
  user_cate_col_hash = tf.feature_column.categorical_column_with_hash_bucket('user_cate_col', hash_bucket_size = 50)



  #one-hot features: "item_id", "sku_id", "user_id", "cate_id", "cate_level1_id", "brand_id"

  item_id = tf.feature_column.indicator_column("item_id")
  sku_id= tf.feature_column.indicator_column("sku_id")
  user_id = tf.feature_column.indicator_column("user_id")
  cate_id = tf.feature_column.indicator_column("cate_id")
  cate_level1_id = tf.feature_column.indicator_column("cate_level1_id")
  brand_id = tf.feature_column.indicator_column("brand_id")


  # Transformations.
  crossed_columns = [
      tf.feature_column.crossed_column(
          ['user_id', 'item_id']),
      tf.feature_column.crossed_column(
          ['user_id', 'cate_id']),
      tf.feature_column.crossed_column(
          ['user_id', 'cate_level1_id']),
      tf.feature_column.crossed_column(
          ['user_id', 'brand_id']),
      tf.feature_column.crossed_column(
          ['user_id', 'disc_hash']),
      tf.feature_column.crossed_column(
          ['user_id', 'price_avg_hash']),
      tf.feature_column.crossed_column(
          ['user_id', 'price_daily_hash'])
  ]


  # Wide columns and deep columns.
  bucket_columns = [disc_hash, price_avg_hash, price_daily_hash, sku_crt_15d_hash, sku_crt_diff_hash, sku_gmv_15d_hash, sku_gmv_diff_hash, sku_ord_15d_hash,
                  sku_ord_diff_hash, sku_sls_15d_hash, sku_sls_diff_hash, user_cate_pv_hash, user_cate_pay_hash, user_brand_pv_hash, user_brand_pay_hash,
                  gmv_xd_sum_hash, pv_xd_sum_hash, user_cate_col_hash
  ]

  onehot_columns = [item_id, sku_id, user_id, cate_id, cate_level1_id, brand_id]

  wide_columns = bucket_columns + onehot_columns + crossed_columns

  dense_columns = [sku_ord_15d, sku_sls_15d, sku_crt_15d, odr_v_xd_sum, sls_v_xd_sum, crt_v_xd_sum, pv_xd_sum,
                   col_v_xd_sum, user_cate_pv, user_cate_crt, user_cate_col, user_cate_pay, user_brand_pv,
                   user_brand_crt, user_brand_col, user_brand_pay, price_avg, price_daily, disc, sku_gmv_15d,
                   sku_ord_diff, sku_gmv_diff, sku_sls_diff, sku_crt_diff, gmv_xd_sum]

  embedding_columns = [tf.feature_column.embedding_column(disc_hash, dimension=5),
                       tf.feature_column.embedding_column(price_avg_hash, dimension=5),
                       tf.feature_column.embedding_column(price_daily_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_crt_15d_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_crt_diff_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_gmv_15d_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_gmv_diff_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_ord_15d_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_ord_diff_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_sls_15d_hash, dimension=5),
                       tf.feature_column.embedding_column(sku_sls_diff_hash, dimension=5),
                       tf.feature_column.embedding_column(user_cate_pv_hash, dimension=5),
                       tf.feature_column.embedding_column(user_cate_pay_hash, dimension=5),
                       tf.feature_column.embedding_column(user_brand_pv_hash, dimension=5),
                       tf.feature_column.embedding_column(user_brand_pay_hash, dimension=5),
                       tf.feature_column.embedding_column(gmv_xd_sum_hash, dimension=5),
                       tf.feature_column.embedding_column(pv_xd_sum_hash, dimension=5),
                       tf.feature_column.embedding_column(user_cate_col_hash, dimension=5),
                       tf.feature_column.embedding_column(item_id, dimension=5),
                       tf.feature_column.embedding_column(sku_id, dimension=5),
                       tf.feature_column.embedding_column(user_id, dimension=5),
                       tf.feature_column.embedding_column(cate_id, dimension=5),
                       tf.feature_column.embedding_column(cate_level1_id, dimension=5),
                       tf.feature_column.embedding_column(brand_id, dimension=5)
                       ]

  deep_columns = dense_columns + embedding_columns

  return wide_columns, deep_columns


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run uv_dataset.py')

  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('label')
    features.pop('ds')          # pop ds
    classes = labels  # binary classification
    return features, classes

  # Extract lines from input files using the Dataset API.
  # need use csv format
  # dataset = tf.data.TextLineDataset(data_file)
  dataset = tf.contrib.data.CsvDataset(data_file, _CSV_COLUMN_TYPE, header=True, na_value='')

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

