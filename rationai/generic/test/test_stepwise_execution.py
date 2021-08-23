import unittest

import rationai.generic.stepwise_execution as tst


class TestToContextKey(unittest.TestCase):
    def test_extract_basic_context_key(self):
        self.assertEqual('context', tst.to_context_key('context.step'))
        self.assertEqual('context2', tst.to_context_key('context2.step2'))

    def test_extract_context_key_from_noncontextual_step(self):
        self.assertEqual('step', tst.to_context_key('step'))

    def test_malformed_step_raises_value_error(self):
        with self.assertRaises(ValueError):
            tst.to_context_key('context.strange.expression')

        with self.assertRaises(ValueError):
            tst.to_context_key('.strange')

        with self.assertRaises(ValueError):
            tst.to_context_key('.strange.part')

        with self.assertRaises(ValueError):
            tst.to_context_key('part.')

        with self.assertRaises(ValueError):
            tst.to_context_key('strange.part.')


class TestStepConfig(unittest.TestCase):
    def setUp(self):
        self.base_definition = dict(
            init=dict(class_id='some_id', config=dict()),
            exec=dict(method='some_method', kwargs=dict())
        )

    def test_existing_step_key_parses_to_object(self):
        step_definitions = dict(step=self.base_definition)
        step_config = tst.StepConfig.from_step_definitions(
            'step', step_definitions
        )
        self.assertIsNotNone(step_config)

    def test_contextual_step(self):
        step_definitions = {'context.step': self.base_definition}
        step_config = tst.StepConfig.from_step_definitions(
            'context.step', step_definitions
        )
        self.assertTrue(step_config.is_contextual_step)

    def test_noncontextual_step(self):
        step_definitions = dict(step=self.base_definition)
        step_config = tst.StepConfig.from_step_definitions(
            'step', step_definitions
        )
        self.assertFalse(step_config.is_contextual_step)

    def test_missing_step_key_parses_to_none(self):
        step_config = tst.StepConfig.from_step_definitions('test', {})
        self.assertIsNone(step_config)

    def test_malformed_step_key_parses_to_none(self):
        step_definitions = {
            'context.strange.expression': self.base_definition,
            '.strange': self.base_definition,
            '.strange.part': self.base_definition,
            'part.': self.base_definition,
            'strange.part.': self.base_definition
        }

        for step_key in step_definitions:
            step_config = tst.StepConfig.from_step_definitions(
                step_key, step_definitions
            )
            self.assertIsNone(step_config)

    def test_parse_init_info_requires_init(self):
        del self.base_definition['init']

        parsed = tst.StepConfig._parse_init_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertIsNone(parsed[0])
        self.assertIsNone(parsed[1])

    def test_parse_init_info_requires_class_id(self):
        del self.base_definition['init']['class_id']

        parsed = tst.StepConfig._parse_init_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertIsNone(parsed[0])
        self.assertIsNone(parsed[1])

    def test_parse_init_info_config_is_optional(self):
        del self.base_definition['init']['config']

        parsed = tst.StepConfig._parse_init_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertEqual('some_id', parsed[0])
        self.assertDictEqual(dict(), parsed[1])

    def test_parse_init_info(self):
        config = dict(param_a=1)
        self.base_definition['init']['class_id'] = 'test_parse_id'
        self.base_definition['init']['config'] = config

        parsed = tst.StepConfig._parse_init_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertEqual('test_parse_id', parsed[0])
        self.assertDictEqual(config, parsed[1])

    def test_parse_exec_info_requires_exec(self):
        del self.base_definition['exec']

        parsed = tst.StepConfig._parse_exec_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertIsNone(parsed[0])
        self.assertIsNone(parsed[1])

    def test_parse_exec_info_requires_method(self):
        del self.base_definition['exec']['method']

        parsed = tst.StepConfig._parse_exec_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertIsNone(parsed[0])
        self.assertIsNone(parsed[1])

    def test_parse_exec_info_kwargs_is_optional(self):
        del self.base_definition['exec']['kwargs']

        parsed = tst.StepConfig._parse_exec_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertEqual('some_method', parsed[0])
        self.assertDictEqual(dict(), parsed[1])

    def test_parse_exec_info(self):
        kwargs = dict(param_b=['test'])
        self.base_definition['exec']['method'] = 'test_parse_method'
        self.base_definition['exec']['kwargs'] = kwargs

        parsed = tst.StepConfig._parse_exec_info('step', self.base_definition)
        self.assertEqual(2, len(parsed))
        self.assertEqual('test_parse_method', parsed[0])
        self.assertDictEqual(kwargs, parsed[1])

    def test_class_id_and_method_are_mandatory(self):
        step_definitions = {
            'without_class_id': self.base_definition.copy(),
            'without_method': self.base_definition.copy()
        }

        del step_definitions['without_class_id']['init']['class_id']

        step_config = tst.StepConfig.from_step_definitions(
            'without_class_id', step_definitions
        )
        self.assertIsNone(step_config)

        del step_definitions['without_class_id']['exec']['method']

        step_config = tst.StepConfig.from_step_definitions(
            'without_method', step_definitions
        )
        self.assertIsNone(step_config)


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
