from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.testing import integration
from official.wide_deep import uv_dataset
from official.wide_deep import uv_main
from official.wide_deep import wide_deep_run_loop

tf.logging.set_verbosity(tf.logging.ERROR)

"""
TEST_INPUT = ('18,Self-emp-not-inc,987,Bachelors,12,Married-civ-spouse,abc,'
              'Husband,zyx,wvu,34,56,78,tsr,<=50K')

TEST_INPUT_VALUES = {
    'age': 18,
    'education_num': 12,
    'capital_gain': 34,
    'capital_loss': 56,
    'hours_per_week': 78,
    'education': 'Bachelors',
    'marital_status': 'Married-civ-spouse',
    'relationship': 'Husband',
    'workclass': 'Self-emp-not-inc',
    'occupation': 'abc',
}
"""

TEST_CSV = os.path.join(os.path.dirname(__file__), 'uv_sampled_test.csv')


class BaseTest(tf.test.TestCase):
    """Tests for Wide Deep model."""

    @classmethod
    def setUpClass(cls):  # pylint: disable=invalid-name
        super(BaseTest, cls).setUpClass()
        uv_main.define_uv_flags()

    def setUp(self):
        # Create temporary CSV file
        self.temp_dir = self.get_temp_dir()

        with tf.gfile.Open(TEST_CSV, "r") as temp_csv:
            test_csv_contents = temp_csv.read()

        # Used for end-to-end tests.
        for fname in [uv_dataset.TRAINING_FILE, uv_dataset.EVAL_FILE]:
            with tf.gfile.Open(os.path.join(self.temp_dir, fname), 'w') as test_csv:
                test_csv.write(test_csv_contents)

    def test_input_fn(self):
        dataset = uv_dataset.input_fn(self.input_csv, 1, False, 1)
        features, labels = dataset.make_one_shot_iterator().get_next()

        with self.test_session() as sess:
            features, labels = sess.run((features, labels))

            # Compare the two features dictionaries.
            """
            for key in TEST_INPUT_VALUES:
                self.assertTrue(key in features)
                self.assertEqual(len(features[key]), 1)
                feature_value = features[key][0]

                # Convert from bytes to string for Python 3.
                if isinstance(feature_value, bytes):
                    feature_value = feature_value.decode()

                self.assertEqual(TEST_INPUT_VALUES[key], feature_value)
            """
            self.assertFalse(labels)

    def build_and_test_estimator(self, model_type):
        """Ensure that model trains and minimizes loss."""
        model = uv_main.build_estimator(
            self.temp_dir, model_type,
            model_column_fn=uv_dataset.build_model_columns,
            inter_op=0, intra_op=0)

        # Train for 1 step to initialize model and evaluate initial loss
        def get_input_fn(num_epochs, shuffle, batch_size):
            def input_fn():
                return uv_dataset.input_fn(
                    TEST_CSV, num_epochs=num_epochs, shuffle=shuffle,
                    batch_size=batch_size)

            return input_fn

        model.train(input_fn=get_input_fn(1, True, 1), steps=1)
        initial_results = model.evaluate(input_fn=get_input_fn(1, False, 1))

        # Train for 100 epochs at batch size 3 and evaluate final loss
        model.train(input_fn=get_input_fn(100, True, 3))
        final_results = model.evaluate(input_fn=get_input_fn(1, False, 1))

        print('%s initial results:' % model_type, initial_results)
        print('%s final results:' % model_type, final_results)

        # Ensure loss has decreased, while accuracy and both AUCs have increased.
        self.assertLess(final_results['loss'], initial_results['loss'])
        self.assertGreater(final_results['auc'], initial_results['auc'])
        self.assertGreater(final_results['auc_precision_recall'],
                           initial_results['auc_precision_recall'])
        self.assertGreater(final_results['accuracy'], initial_results['accuracy'])

    def test_wide_deep_estimator_training(self):
        self.build_and_test_estimator('wide_deep')

    def test_end_to_end_wide(self):
        integration.run_synthetic(
            main=uv_main.main, tmp_root=self.get_temp_dir(),
            extra_flags=[
                '--data_dir', self.get_temp_dir(),
                '--model_type', 'wide',
                '--download_if_missing=false'
            ],
            synth=False, max_train=None)

    def test_end_to_end_deep(self):
        integration.run_synthetic(
            main=uv_main.main, tmp_root=self.get_temp_dir(),
            extra_flags=[
                '--data_dir', self.get_temp_dir(),
                '--model_type', 'deep',
                '--download_if_missing=false'
            ],
            synth=False, max_train=None)

    def test_end_to_end_wide_deep(self):
        integration.run_synthetic(
            main=uv_main.main, tmp_root=self.get_temp_dir(),
            extra_flags=[
                '--data_dir', self.get_temp_dir(),
                '--model_type', 'wide_deep',
                '--download_if_missing=false'
            ],
            synth=False, max_train=None)


if __name__ == '__main__':
    tf.test.main()