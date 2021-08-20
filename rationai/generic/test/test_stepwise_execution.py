import unittest

import rationai.generic.stepwise_execution as tst


class TestExtractContextKey(unittest.TestCase):
    def test_extract_basic_context_key(self):
        self.assertEqual('context', tst.extract_context_key('context.step'))
        self.assertEqual('context2', tst.extract_context_key('context2.step2'))

    def test_extract_context_key_from_noncontextual_step(self):
        self.assertEqual('step', tst.extract_context_key('step'))

    def test_malformed_step_raises_value_error(self):
        with self.assertRaises(ValueError):
            tst.extract_context_key('context.strange.expression')

        with self.assertRaises(ValueError):
            tst.extract_context_key('.strange')

        with self.assertRaises(ValueError):
            tst.extract_context_key('.strange.part')

        with self.assertRaises(ValueError):
            tst.extract_context_key('part.')

        with self.assertRaises(ValueError):
            tst.extract_context_key('strange.part.')


class TestIsStepContextual(unittest.TestCase):
    def test_contextual_step(self):
        self.assertTrue(tst.is_step_contextual('context.step'))

    def test_noncontextual_step(self):
        self.assertFalse(tst.is_step_contextual('step'))

    def test_malformed_step_raises_value_error(self):
        with self.assertRaises(ValueError):
            tst.extract_context_key('context.strange.expression')

        with self.assertRaises(ValueError):
            tst.extract_context_key('.strange')

        with self.assertRaises(ValueError):
            tst.extract_context_key('.strange.part')

        with self.assertRaises(ValueError):
            tst.extract_context_key('part.')

        with self.assertRaises(ValueError):
            tst.extract_context_key('strange.part.')


class TestStepInterfaceSubclassing(unittest.TestCase):
    def test_cannot_initialize(self):
        with self.assertRaises(TypeError):
            tst.StepInterface()

    def test_cannot_run_from_params(self):
        with self.assertRaises(NotImplementedError):
            tst.StepInterface.from_params({}, {})

    def test_cannot_run_continue_from_run(self):
        with self.assertRaises(NotImplementedError):
            tst.StepInterface.continue_from_run(None)

    def test_cannot_subclass_without_from_params(self):
        class StepInterfaceImpl(tst.StepInterface):
            def continue_from_run(self):
                pass

        self.assertFalse(issubclass(StepInterfaceImpl, tst.StepInterface))

    def test_cannot_subclass_without_continue_from_run(self):
        class StepInterfaceImpl(tst.StepInterface):
            @classmethod
            def from_params(cls, self_config, params):
                pass

        self.assertFalse(issubclass(StepInterfaceImpl, tst.StepInterface))

    def test_cannot_subclass_with_wrong_signature(self):
        class StepInterfaceImpl(tst.StepInterface):
            def continue_from_run(self):
                pass

            @classmethod
            def from_params(cls, params):
                pass

        class StepInterfaceImpl2(tst.StepInterface):
            def continue_from_run(self, some_param):
                pass

            @classmethod
            def from_params(cls, self_config, params):
                pass

        self.assertFalse(issubclass(StepInterfaceImpl, tst.StepInterface))
        self.assertFalse(issubclass(StepInterfaceImpl2, tst.StepInterface))

    def test_can_subclass_with_constraints_satisfied(self):
        class StepInterfaceImpl(tst.StepInterface):
            @classmethod
            def from_params(cls, self_config, params):
                pass

            def continue_from_run(self):
                pass

        self.assertTrue(issubclass(StepInterfaceImpl, tst.StepInterface))
