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
