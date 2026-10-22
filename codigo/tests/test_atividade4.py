"""Regressões da Atividade 4; executar com python -m unittest discover -s tests."""
import unittest

import numpy as np

from examples.sample_data_utils import get_tabular_sample_data, set_abnormal_data


class DataTests(unittest.TestCase):
    def test_first_interval_uses_duration_instead_of_absolute_end(self):
        x, y = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000),
                                 features=[0], start_point=200, magnitude=7)
        self.assertTrue(np.all(y[200:213] == 1))
        self.assertTrue(np.all(x[200:213, 0] > 0))
        self.assertEqual(y[213], 0)

    def test_explicit_zero_start_is_respected(self):
        _, y = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), start_point=0)
        self.assertTrue(np.all(y[:13] == 1))
        self.assertEqual(y[13], 0)

    def test_default_start_creates_nonempty_interval(self):
        _, y = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000))
        self.assertTrue(np.all(y[13:26] == 1))

    def test_other_features_are_unchanged(self):
        x, _ = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), features=[0, 1])
        np.testing.assert_array_equal(x[:, 2:], 0)

    def test_overlapping_anomalies_keep_positive_label(self):
        x, y = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), start_point=200)
        _, y = set_abnormal_data(x, 13, y, start_point=205)
        self.assertTrue(np.all(y[200:218] == 1))

    def test_invalid_interval_fails_instead_of_silently_doing_nothing(self):
        with self.assertRaises(ValueError):
            set_abnormal_data(np.zeros((1000, 6)), 0, np.zeros(1000))
        with self.assertRaises(ValueError):
            set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), start_point=-1)
        with self.assertRaises(ValueError):
            set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(999))
        with self.assertRaises(ValueError):
            set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), start_point=995)

    def test_chronological_holdout_and_training_are_separate(self):
        xtr, ytr, xte, yte, meta = get_tabular_sample_data(return_metadata=True)
        self.assertEqual(xtr.shape, (3000, 6))
        self.assertEqual(xte.shape, (2000, 6))
        self.assertEqual(int(ytr.sum()), 0)
        self.assertTrue(0 < yte.sum() < len(yte))
        self.assertLess(max(meta['train_indices']), min(meta['test_indices']))
        self.assertTrue(np.all(np.diff(meta['test_indices']) == 1))

    def test_repeated_calls_are_identical_and_do_not_change_numpy_rng(self):
        np.random.seed(91)
        expected = np.random.random()
        np.random.seed(91)
        a = get_tabular_sample_data()
        self.assertEqual(np.random.random(), expected)
        b = get_tabular_sample_data()
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first, second)

    def test_legacy_partition_is_explicit_and_metadata_matches(self):
        xtr, ytr, xte, yte, meta = get_tabular_sample_data(split_mode='legacy', return_metadata=True)
        self.assertEqual(len(xtr), len(meta['train_indices']))
        self.assertEqual(len(xte), len(meta['test_indices']))
        self.assertEqual(yte.sum(), len(yte) // 2)
        self.assertEqual(ytr.sum(), 0)
        self.assertFalse(set(meta['train_indices']) & set(meta['test_indices']))
        self.assertEqual(len(get_tabular_sample_data(split_mode='legacy')), 4)

    def test_unknown_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            get_tabular_sample_data(split_mode='shuffle')


class HitachiIntegrationTests(unittest.TestCase):
    def test_reported_metrics_match_independent_confusion_counts(self):
        from atividade4.experimento import audit_original, metrics
        result = metrics(np.array([0, 0, 1, 1]), np.array([.1, .9, .8, .7]), .75)
        self.assertEqual([result[k] for k in ['tn','fp','fn','tp']], [1,1,1,1])
        for key in ['f1','precision','recall','far','mar','roc_auc']:
            self.assertEqual(result[key], .5)
        audit = audit_original()
        self.assertEqual(audit['original_positive_rows_200_213'], 0)
        self.assertEqual(audit['corrected_positive_rows_200_213'], 13)

    def test_score_direction_and_channel_adaptation(self):
        import torch
        from torch import nn

        from mtsa.predictors.hitachi_score import HitachiScore
        class ZeroModel(nn.Module):
            input_dim = 6
            def forward(self, x):
                return torch.zeros_like(x)
        function = HitachiScore(ZeroModel()).eval()
        x = torch.stack([torch.ones(3, 6), torch.full((3, 6), 2.)])
        torch.testing.assert_close(function(x), torch.tensor([1., 4.]))
        torch.testing.assert_close(function(x.unsqueeze(-1)), torch.tensor([1., 4.]))
        with self.assertRaises(ValueError):
            function(torch.ones(1, 3, 7))

    def test_trainer_and_predictor_execute_with_one_score_per_row(self):
        import torch

        from mtsa.losses.mse_loss import MSELoss
        from mtsa.models.hitachi import Hitachi
        from mtsa.predictors.hitachi_score import HitachiScore
        from mtsa.predictors.standard_predictor import StandardPredictor
        from mtsa.trainers.unsupervised_trainer import UnsupervisedTrainer
        from mtsa.utils.data import get_dataloader
        torch.manual_seed(9)
        x = np.linspace(-1, 1, 96, dtype=np.float32).reshape(16, 6)
        y = np.zeros(16, dtype=int)
        dl = get_dataloader(x, y, window_size=1, stride_size=1, batch_size=8, drop_last=False)
        model = Hitachi(6)
        UnsupervisedTrainer().fit(model, torch.optim.Adam(model.parameters()), MSELoss(), dl, epochs=1)
        function = HitachiScore(model).eval()
        scores, labels = StandardPredictor().predict(function, dl, pointwise=True, device='cpu')
        self.assertEqual(scores.shape, (16, 1))
        self.assertEqual(labels.shape, (16, 1))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
