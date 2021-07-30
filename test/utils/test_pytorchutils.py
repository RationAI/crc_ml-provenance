import unittest

import torch

from rationai.utils import pytorchutils as tst


class TestGetPytorchLoss(unittest.TestCase):
    def test_name_none_returns_none(self):
        self.assertIsNone(tst.get_pytorch_loss(None))

    def test_binary_crossentropy_returns_loss_object(self):
        self.assertIsInstance(
            tst.get_pytorch_loss('BinaryCrossentropy'), torch.nn.BCELoss
        )

    def test_unknown_name_returns_none(self):
        self.assertIsNone(tst.get_pytorch_loss('unknown_loss'))


class TestGetPytorchOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = {'lr': 1e-7, 'momentum': .9, 'epsilon': 1., 'rho': .9}

        class TestNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 1)
                self.dropout = torch.nn.Dropout2d(0.25)

            def forward(self, inp):
                return self.dropout(self.fc(inp))

        self.net = TestNet()

    def test_name_none_returns_none(self):
        self.assertIsNone(tst.get_pytorch_optimizer(None, self.config))

    def test_rmsprop_returns_rmsprop_optimizer(self):
        optim_gen = tst.get_pytorch_optimizer('RMSProp', self.config)

        optimizer = optim_gen(self.net.parameters())
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertIsInstance(optimizer, torch.optim.RMSprop)

    def test_rmsprop_parameters_are_passed_correctly(self):
        optim_gen = tst.get_pytorch_optimizer('RMSProp', self.config)
        optimizer = optim_gen(self.net.parameters())
        for param_group in optimizer.param_groups:
            self.assertEqual(self.config['lr'], param_group['lr'])
            self.assertEqual(self.config['epsilon'], param_group['eps'])
            self.assertEqual(self.config['rho'], param_group['weight_decay'])
            self.assertEqual(self.config['momentum'], param_group['momentum'])

    def test_rmsprop_invalid_config_raises_value_error(self):
        with self.assertRaises(ValueError):
            tst.get_pytorch_optimizer('RMSProp', {})

        with self.assertRaises(ValueError):
            tst.get_pytorch_optimizer('RMSProp', None)

        config = dict(self.config)
        del config['rho']

        with self.assertRaises(ValueError):
            tst.get_pytorch_optimizer('RMSProp', config)

    def test_unknown_name_returns_none(self):
        self.assertIsNone(tst.get_pytorch_optimizer('unknown_opt', {}))


class TestGetPytorchRegularizer(unittest.TestCase):
    def test_name_none_returns_none(self):
        self.assertIsNone(tst.get_pytorch_regularizer(None, {'l1': 1}))

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

    def test_l2_invalid_config_raises_value_error(self):
        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L2', {})

        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L2', None)

        with self.assertRaises(ValueError):
            tst.get_pytorch_regularizer('L2', {'l1': 1})

    def test_unknown_name_returns_none(self):
        self.assertIsNone(tst.get_pytorch_regularizer('unknown_reg', {}))
