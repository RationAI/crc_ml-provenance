import unittest

import torch

from rationai.utils import pytorchutils as tst


class TestGetPytorchRegularizer(unittest.TestCase):
    def test_name_none_returns_none(self):
        self.assertIsNone(tst.get_pytorch_regularizer(None, None))

    def test_l1_returns_l1_regularizer(self):
        regularizer = tst.get_pytorch_regularizer('L1', {'l1': 1})
        reg_param = regularizer(torch.tensor([0.1, 0.2, 3]))

        self.assertIsInstance(reg_param, torch.Tensor)
        self.assertEqual(3.3, reg_param)

        tensor = torch.tensor([-1., 1.])
        self.assertEqual(2, regularizer(tensor))

    def test_l2_returns_l2_regularizer(self):
        regularizer = tst.get_pytorch_regularizer('L2', {'l2': 1})
        reg_param = regularizer(torch.tensor([0.1, 0.2, 3]))

        self.assertIsInstance(reg_param, torch.Tensor)
        self.assertEqual(9.05, reg_param)

        tensor = torch.tensor([-1., 1.])
        self.assertEqual(2, regularizer(tensor))

    def test_l1_regularization_coefficient_affects_result(self):
        regularizer = tst.get_pytorch_regularizer('L1', {'l1': .1})
        self.assertEqual(.33, regularizer(torch.tensor([0.1, 0.2, 3])))

        regularizer = tst.get_pytorch_regularizer('L1', {'l1': .2})
        self.assertEqual(.66, regularizer(torch.tensor([0.1, 0.2, 3])))

        regularizer = tst.get_pytorch_regularizer('L1', {'l1': -1})
        self.assertEqual(-3.3, regularizer(torch.tensor([0.1, 0.2, 3])))

    def test_l2_regularization_coefficient_affects_result(self):
        regularizer = tst.get_pytorch_regularizer('L2', {'l2': .1})
        # deal with rounding problems
        value = regularizer(torch.tensor([0.1, 0.2, 3])).item()
        self.assertAlmostEqual(.905, value, 3)

        regularizer = tst.get_pytorch_regularizer('L2', {'l2': .2})
        # deal with rounding problems again
        value = regularizer(torch.tensor([0.1, 0.2, 3])).item()
        self.assertAlmostEqual(1.81, value, 2)

        regularizer = tst.get_pytorch_regularizer('L2', {'l2': -1})
        self.assertEqual(-9.05, regularizer(torch.tensor([0.1, 0.2, 3])))

    def test_l1_invalid_config_raises_value_error(self):
        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L1', {})

        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L1', None)

        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L1', {'l2': 1})

    def test_l2_missing_config_raises_value_error(self):
        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L2', {})

        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L2', None)

        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L2', {'l1': 1})

    def test_unknown_name_returns_none(self):
        self.assertIsNone(tst.get_pytorch_regularizer('unknown_reg', {}))
