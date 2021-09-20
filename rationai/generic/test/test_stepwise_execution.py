import unittest

import rationai.generic.stepwise_execution as tst


class StepInterfaceDummy(tst.StepInterface):
    @classmethod
    def from_params(
            cls,
            self_config: dict,
            params: dict,
            dir_structure: tst.DirStructure
    ):
        return cls(**self_config)

    def continue_from_run(self):
        pass

    @staticmethod
    def run(fail=False):
        if fail:
            raise IndexError()


class StepInterfaceErrorRaisingDummy(tst.StepInterface):
    @classmethod
    def from_params(
            cls,
            self_config: dict,
            params: dict,
            dir_structure: tst.DirStructure
    ):
        raise ValueError()

    def continue_from_run(self):
        pass


class StepInterfaceWithParamsDummy(tst.StepInterface):

    def __init__(self, param_id):
        self.id = param_id

    @classmethod
    def from_params(
            cls,
            self_config: dict,
            params: dict,
            dir_structure: tst.DirStructure
    ):
        return cls(**self_config)

    def continue_from_run(self):
        pass

    def run(self, state_number):
        pass


class TestInitializeStep(unittest.TestCase):

    def test_cannot_initialize_not_subclass_of_step_interface(self):
        params_dummy = {}

        class StepConfigDummy:
            class_id = 'collections.OrderedDict'
            step_key = 'test_step'
            init_params = {}

        class DirStructDummy:
            pass

        self.assertIsNone(
            tst.initialize_step(params_dummy, StepConfigDummy(), DirStructDummy())
        )

    def test_can_initialize_step_interface(self):
        params_dummy = {}

        class StepConfigDummy:
            class_id = 'rationai.generic.test.test_stepwise_execution.StepInterfaceDummy'
            step_key = 'test_step'
            init_params = {}

        class DirStructDummy:
            pass

        self.assertIsNotNone(
            tst.initialize_step(params_dummy, StepConfigDummy(), DirStructDummy())
        )

    def test_return_none_on_error(self):
        params_dummy = {}

        class StepConfigDummy:
            class_id = 'rationai.generic.test.test_stepwise_execution.StepInterfaceErrorRaisingDummy'
            step_key = 'test_step'
            init_params = {}

        class DirStructDummy:
            pass

        self.assertIsNone(
            tst.initialize_step(params_dummy, StepConfigDummy(), DirStructDummy())
        )


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


class TestStepExecutor(unittest.TestCase):
    def setUp(self):
        class DirStructDummy:
            pass

        self.dir_struct = DirStructDummy(),

        self.valid_dummy_with_params_config = dict(
            init=dict(
                class_id='rationai.generic.test.test_stepwise_execution.StepInterfaceWithParamsDummy',
                config=dict(param_id=1)
            ),
            exec=dict(
                method='run',
                kwargs=dict(state_number=42)
            )
        )

        self.valid_dummy_no_fail_config = dict(
            init=dict(
                class_id='rationai.generic.test.test_stepwise_execution.StepInterfaceDummy',
                config=dict()
            ),
            exec=dict(
                method='run'
            )
        )

        self.valid_dummy_fail_config = dict(
            init=dict(
                class_id='rationai.generic.test.test_stepwise_execution.StepInterfaceDummy',
                config=dict()
            ),
            exec=dict(
                method='run',
                # Make the run fail
                kwargs=dict(fail=True)
            )
        )

        self.invalid_dummy_no_exec_method_config = dict(
            init=dict(
                class_id='rationai.generic.test.test_stepwise_execution.StepInterfaceDummy',
                config=dict(id=1)
            ),
            exec=dict(
                # the method config option is mandatory
                # method='run'
            )
        )

        self.invalid_dummy_bad_params_config = dict(
            init=dict(
                class_id='rationai.generic.test.test_stepwise_execution.StepInterfaceDummy',
                config=dict(nonexisting_param=1)
            ),
            exec=dict(
                method='run'
            )
        )

    def test_proper_initialization(self):
        executor = tst.StepExecutor(
            step_keys=['test_key'],
            step_definitions=dict(some_key=dict()),
            params=dict(param_a=1),
            dir_structure=self.dir_struct
        )

        self.assertEqual(-1, executor.current_step_idx)
        self.assertListEqual(['test_key'], executor.step_keys)
        self.assertDictEqual(dict(some_key=dict()), executor.step_definitions)
        self.assertDictEqual(dict(param_a=1), executor.params)
        self.assertIs(self.dir_struct, executor.dir_structure)
        self.assertDictEqual(dict(), executor.context)

    def test_next_step_key(self):
        executor = tst.StepExecutor(
            step_keys=['test_key'],
            step_definitions=dict(some_key=dict()),
            params=dict(param_a=1),
            dir_structure=self.dir_struct
        )
        self.assertEqual('test_key', executor.next_step_key())

        executor = tst.StepExecutor(
            step_keys=[],
            step_definitions=dict(some_key=dict()),
            params=dict(param_a=1),
            dir_structure=self.dir_struct
        )
        self.assertIsNone(executor.next_step_key())

    def test_run_next_moves_index(self):
        executor = tst.StepExecutor(
            step_keys=['test_step'],
            step_definitions=dict(
                test_step=self.valid_dummy_with_params_config
            ),
            params=dict(),
            dir_structure=self.dir_struct
        )
        self.assertFalse(executor.run_next())
        self.assertEqual(0, executor.current_step_idx)

        self.assertFalse(executor.run_next())
        self.assertEqual(0, executor.current_step_idx)

    def test_run_next_can_run_multiple_steps(self):
        executor = tst.StepExecutor(
            step_keys=['test_step', 'test_step_1'],
            step_definitions=dict(
                test_step=self.valid_dummy_with_params_config,
                test_step_1=self.valid_dummy_with_params_config
            ),
            params=dict(),
            dir_structure=self.dir_struct
        )
        self.assertTrue(executor.run_next())
        self.assertEqual(0, executor.current_step_idx)

        self.assertFalse(executor.run_next())
        self.assertEqual(1, executor.current_step_idx)

    def test_run_next_holds_context(self):
        executor = tst.StepExecutor(
            step_keys=['ctx.test_step', 'test_step_1', 'ctx.test_step_2', 'test_step_3'],
            step_definitions={
                'ctx.test_step': self.valid_dummy_with_params_config,
                'test_step_1': self.valid_dummy_no_fail_config,
                # Watch out for the discrepancy:
                # class_id - StepInterfaceDummy
                # run kwargs - state_number
                # because a StepInterfaceWithMemoryDummy object is saved in
                # the context, class_id gets ignored
                # TODO: Potentially fix this to fail explicitly in this case
                'ctx.test_step_2': dict(
                    init=dict(
                        class_id='rationai.generic.test.test_stepwise_execution.StepInterfaceDummy',
                        config=dict(param_id=2)
                    ),
                    exec=dict(
                        method='run',
                        kwargs=dict(state_number=42)
                    )
                ),
                'test_step_3': self.valid_dummy_no_fail_config,
            },
            params=dict(),
            dir_structure=self.dir_struct
        )
        self.assertDictEqual({}, executor.context)
        self.assertTrue(executor.run_next())
        self.assertEqual(0, executor.current_step_idx)
        self.assertTrue('ctx' in executor.context)
        self.assertIsInstance(executor.context['ctx'], StepInterfaceWithParamsDummy)

        self.assertTrue(executor.run_next())
        self.assertEqual(1, executor.current_step_idx)
        self.assertTrue('ctx' in executor.context)
        self.assertIsInstance(executor.context['ctx'], StepInterfaceWithParamsDummy)

        self.assertTrue(executor.run_next())
        self.assertEqual(2, executor.current_step_idx)
        self.assertFalse('ctx' in executor.context)

        self.assertFalse(executor.run_next())
        self.assertEqual(3, executor.current_step_idx)

    def test_error_in_step_execution_fails_run(self):
        executor = tst.StepExecutor(
            step_keys=['test_step_0', 'test_step_1', 'test_step_2'],
            step_definitions={
                'test_step_0': self.valid_dummy_no_fail_config,
                'test_step_1': self.valid_dummy_fail_config,
                'test_step_2': self.valid_dummy_no_fail_config
            },
            params=dict(),
            dir_structure=self.dir_struct
        )
        self.assertTrue(executor.run_next())
        self.assertEqual(0, executor.current_step_idx)

        self.assertFalse(executor.run_next())
        self.assertEqual(1, executor.current_step_idx)

    def test_invalid_step_config_disables_run(self):
        executor = tst.StepExecutor(
            step_keys=['test_step_0', 'test_step_1'],
            step_definitions={
                'test_step_0': self.invalid_dummy_no_exec_method_config,
                'test_step_1': self.valid_dummy_no_fail_config
            },
            params=dict(),
            dir_structure=self.dir_struct
        )

        self.assertFalse(executor.run_next())

    def test_invalid_step_instance(self):
        executor = tst.StepExecutor(
            step_keys=['test_step_0', 'test_step_1'],
            step_definitions={
                'test_step_0': self.invalid_dummy_bad_params_config,
                'test_step_1': self.valid_dummy_no_fail_config
            },
            params=dict(),
            dir_structure=self.dir_struct
        )

        self.assertFalse(executor.run_next())

    def test_run_all(self):
        executor = tst.StepExecutor(
            step_keys=['test_step_0', 'test_step_1', 'test_step_2'],
            step_definitions={
                'test_step_0': self.valid_dummy_no_fail_config,
                'test_step_1': self.valid_dummy_no_fail_config,
                'test_step_2': self.valid_dummy_no_fail_config
            },
            params=dict(),
            dir_structure=self.dir_struct
        )

        executor.run_all()
        self.assertIsNone(executor.next_step_key())
        self.assertEqual(2, executor.current_step_idx)

    def test_run_all_stops_on_fail(self):
        executor = tst.StepExecutor(
            step_keys=['test_step_0', 'test_step_1', 'test_step_2'],
            step_definitions={
                'test_step_0': self.valid_dummy_no_fail_config,
                'test_step_1': self.valid_dummy_fail_config,
                'test_step_2': self.valid_dummy_no_fail_config
            },
            params=dict(),
            dir_structure=self.dir_struct
        )

        executor.run_all()
        self.assertEqual(1, executor.current_step_idx)


class TestStepInterfaceSubclassing(unittest.TestCase):
    def test_cannot_initialize(self):
        with self.assertRaises(TypeError):
            tst.StepInterface()

    def test_cannot_run_from_params(self):
        with self.assertRaises(NotImplementedError):
            tst.StepInterface.from_params({}, {}, None)

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
            def from_params(cls, self_config, params, dir_structure):
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
            def from_params(cls, self_config, params, dir_structure):
                pass

        self.assertFalse(issubclass(StepInterfaceImpl, tst.StepInterface))
        self.assertFalse(issubclass(StepInterfaceImpl2, tst.StepInterface))

    def test_can_subclass_with_constraints_satisfied(self):
        class StepInterfaceImpl(tst.StepInterface):
            @classmethod
            def from_params(cls, self_config, params, dir_structure):
                pass

            def continue_from_run(self):
                pass

        self.assertTrue(issubclass(StepInterfaceImpl, tst.StepInterface))
