import abc
import collections
import unittest

from rationai.utils import utils as tst


class TestCallableHasSignature(unittest.TestCase):
    def test_function_empty_signature(self):
        def f_test():
            pass

        self.assertTrue(tst.callable_has_signature(f_test, []))
        self.assertFalse(tst.callable_has_signature(f_test, ['param']))

    def test_method_empty_signature(self):
        class TestClass:
            def method(self):
                pass

        test_object = TestClass()

        self.assertTrue(tst.callable_has_signature(test_object.method, []))
        self.assertFalse(
            tst.callable_has_signature(test_object.method, ['param'])
        )

    def test_function_proper_signature_passes(self):
        def f_test(param):
            pass

        def f_test_2(param, param2):
            pass

        self.assertTrue(tst.callable_has_signature(f_test, ['param']))
        self.assertTrue(
            tst.callable_has_signature(f_test_2, ['param', 'param2'])
        )

    def test_method_proper_signature_passes(self):
        class TestClass:
            def method(self, param):
                pass

            def method_2(self, param, param2):
                pass

        test_object = TestClass()

        self.assertTrue(
            tst.callable_has_signature(test_object.method, ['param'])
        )
        self.assertTrue(
            tst.callable_has_signature(
                test_object.method_2, ['param', 'param2']
            )
        )

    def test_function_other_signature_fails(self):
        def f_test(param):
            pass

        def f_test_2(param, param2):
            pass

        self.assertFalse(tst.callable_has_signature(f_test, []))
        self.assertFalse(tst.callable_has_signature(f_test, ['different']))
        self.assertFalse(
            tst.callable_has_signature(f_test, ['param', 'other'])
        )

        self.assertFalse(tst.callable_has_signature(f_test_2, []))
        self.assertFalse(tst.callable_has_signature(f_test_2, ['param']))
        self.assertFalse(
            tst.callable_has_signature(f_test_2, ['param', 'other'])
        )

    def test_method_other_signature_passes(self):
        class TestClass:
            def method(self, param):
                pass

            def method_2(self, param, param2):
                pass

        test_object = TestClass()

        self.assertFalse(tst.callable_has_signature(test_object.method, []))
        self.assertFalse(
            tst.callable_has_signature(test_object.method, ['different'])
        )
        self.assertFalse(
            tst.callable_has_signature(test_object.method, ['param', 'other'])
        )

        self.assertFalse(tst.callable_has_signature(test_object.method_2, []))
        self.assertFalse(
            tst.callable_has_signature(test_object.method_2, ['param'])
        )
        self.assertFalse(
            tst.callable_has_signature(
                test_object.method_2, ['param', 'other']
            )
        )

    def test_non_callable_raises_typeerror(self):
        with self.assertRaises(TypeError):
            tst.callable_has_signature('string', [])


class TestClassHasClassmethod(unittest.TestCase):
    def test_class_without_attribute_fails(self):
        class TestClass:
            pass

        self.assertFalse(tst.class_has_classmethod(TestClass, 'test_method'))

    def test_class_with_noncallable_attribute_fails(self):
        class TestClass1:
            test_method = None

        class TestClass2:
            test_method = []

        class TestClass3:
            test_method = 42

        self.assertFalse(tst.class_has_classmethod(TestClass1, 'test_method'))
        self.assertFalse(tst.class_has_classmethod(TestClass2, 'test_method'))
        self.assertFalse(tst.class_has_classmethod(TestClass3, 'test_method'))

    def test_class_with_method_not_classmethod_fails(self):
        def test_function():
            return None

        class TestClass1:
            test_method = test_function

        class TestClass2:
            def test_method(self):
                return None

        class TestClass3:
            @staticmethod
            def test_method():
                return None

        self.assertFalse(tst.class_has_classmethod(TestClass1, 'test_method'))
        self.assertFalse(tst.class_has_classmethod(TestClass2, 'test_method'))
        self.assertFalse(tst.class_has_classmethod(TestClass3, 'test_method'))

    def test_class_with_classmethod_passes(self):
        class TestClass:
            @classmethod
            def test_method(cls):
                return None

        self.assertTrue(tst.class_has_classmethod(TestClass, 'test_method'))


class TestClassHasMethod(unittest.TestCase):
    def test_class_without_attribute_fails(self):
        class TestClass:
            pass

        self.assertFalse(tst.class_has_method(TestClass, 'test_method'))

    def test_class_with_noncallable_attribute_fails(self):
        class TestClass1:
            test_method = None

        class TestClass2:
            test_method = []

        class TestClass3:
            test_method = 42

        self.assertFalse(tst.class_has_method(TestClass1, 'test_method'))
        self.assertFalse(tst.class_has_method(TestClass2, 'test_method'))
        self.assertFalse(tst.class_has_method(TestClass3, 'test_method'))

    def test_class_with_method_passes(self):
        def test_function():
            return None

        class TestClass1:
            test_method = test_function

        class TestClass2:
            def test_method(self):
                return None

        class TestClass3:
            @staticmethod
            def test_method():
                return None

        class TestClass4:
            @classmethod
            def test_method(cls):
                return None

        self.assertTrue(tst.class_has_method(TestClass1, 'test_method'))
        self.assertTrue(tst.class_has_method(TestClass2, 'test_method'))
        self.assertTrue(tst.class_has_method(TestClass3, 'test_method'))
        self.assertTrue(tst.class_has_method(TestClass4, 'test_method'))


class TestClassHasNonabstractMethod(unittest.TestCase):
    def test_class_without_attribute_fails(self):
        class TestClass:
            pass

        self.assertFalse(
            tst.class_has_nonabstract_method(TestClass, 'test_method')
        )

    def test_class_with_noncallable_attribute_fails(self):
        class TestClass1:
            test_method = None

        class TestClass2:
            test_method = []

        class TestClass3:
            test_method = 42

        self.assertFalse(
            tst.class_has_nonabstract_method(TestClass1, 'test_method')
        )
        self.assertFalse(
            tst.class_has_nonabstract_method(TestClass2, 'test_method')
        )
        self.assertFalse(
            tst.class_has_nonabstract_method(TestClass3, 'test_method')
        )

    def test_class_with_nonabstract_method_passes(self):
        def test_function():
            return None

        class TestClass1:
            test_method = test_function

        class TestClass2:
            def test_method(self):
                return None

        class TestClass3:
            @staticmethod
            def test_method():
                return None

        class TestClass4:
            @classmethod
            def test_method(cls):
                return None

        self.assertTrue(tst.class_has_nonabstract_method(TestClass1, 'test_method'))
        self.assertTrue(tst.class_has_nonabstract_method(TestClass2, 'test_method'))
        self.assertTrue(tst.class_has_nonabstract_method(TestClass3, 'test_method'))
        self.assertTrue(tst.class_has_nonabstract_method(TestClass4, 'test_method'))

    def test_class_with_abstract_method_fails(self):
        class TestClass1:
            @abc.abstractmethod
            def test_method(self):
                return None

        class TestClass2:
            @staticmethod
            @abc.abstractmethod
            def test_method():
                return None

        class TestClass3:
            @classmethod
            @abc.abstractmethod
            def test_method(cls):
                return None

        self.assertFalse(
            tst.class_has_nonabstract_method(TestClass1, 'test_method')
        )
        self.assertFalse(
            tst.class_has_nonabstract_method(TestClass2, 'test_method')
        )
        self.assertFalse(
            tst.class_has_nonabstract_method(TestClass3, 'test_method')
        )


class TestLoadClass(unittest.TestCase):
    def test_existing_class_gets_loaded(self):
        loaded = tst.load_class('collections.Counter')
        self.assertIs(loaded, collections.Counter)

        loaded = tst.load_class('collections.OrderedDict')
        self.assertIs(loaded, collections.OrderedDict)

    def test_nonexistent_module_raises_import_error(self):
        with self.assertRaises(ImportError):
            tst.load_class('nonexistentmodule.OrderedDict')

    def test_nonexistent_class_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            tst.load_class('collections.NonexistentClass')


class TestParseModuleAndClassString(unittest.TestCase):
    def test_returns_two_strings(self):
        result_a, result_b = tst.parse_module_and_class_string('module.Class')
        self.assertIsInstance(result_a, str)
        self.assertIsInstance(result_b, str)

    def test_empty_descriptor(self):
        result_a, result_b = tst.parse_module_and_class_string('')
        self.assertEqual('', result_a)
        self.assertEqual('', result_b)

    def test_parses_module_name_properly(self):
        result_a, result_b = tst.parse_module_and_class_string('module.Class')
        self.assertEqual('module', result_a)

        result_a, result_b = tst.parse_module_and_class_string('complex.module.Class')
        self.assertEqual('complex.module', result_a)

    def test_parses_class_name_properly(self):
        result_a, result_b = tst.parse_module_and_class_string('module.Class')
        self.assertEqual('Class', result_b)

        result_a, result_b = tst.parse_module_and_class_string('module.Classy')
        self.assertEqual('Classy', result_b)

        result_a, result_b = tst.parse_module_and_class_string('complex.module.Classiest')
        self.assertEqual('Classiest', result_b)

    def test_class_name_with_dot_only(self):
        result_a, result_b = tst.parse_module_and_class_string('.Class')
        self.assertEqual('', result_a)
        self.assertEqual('Class', result_b)

        result_a, result_b = tst.parse_module_and_class_string('Class.')
        self.assertEqual('', result_a)
        self.assertEqual('Class', result_b)

    def test_complex_module_with_surrounding_dots(self):
        result_a, result_b = tst.parse_module_and_class_string('.test.complex.module.Class.')
        self.assertEqual('test.complex.module', result_a)
        self.assertEqual('Class', result_b)


class TestRunClassmethod(unittest.TestCase):
    def test_classmethod_runs(self):
        class TestClass:
            @classmethod
            def test_method(cls):
                return 2

            @classmethod
            def test_method_subtract(cls, a, b):
                return a - b

        self.assertEqual(2, tst.run_classmethod(TestClass, 'test_method', {}))
        self.assertEqual(
            -3,
            tst.run_classmethod(TestClass, 'test_method_subtract', dict(a=1, b=4))
        )

    def test_non_classmethod_raises_attribute_error(self):
        class TestClass:
            def __init__(self):
                self.return_val = -42

            def test_method(self):
                return self.return_val

        with self.assertRaises(AttributeError):
            tst.run_classmethod(TestClass, 'test_method', {})

    def test_nonexistent_method_raises_attribute_error(self):
        class TestClass:
            attribute = 3

        with self.assertRaises(AttributeError):
            tst.run_classmethod(TestClass, 'attribute', {})

        with self.assertRaises(AttributeError):
            tst.run_classmethod(TestClass, 'nonexistent_method', {})

    def test_raised_error_gets_propagated(self):
        class TestClass:
            @classmethod
            def raising_method(cls):
                raise TypeError

        with self.assertRaises(TypeError):
            tst.run_classmethod(TestClass, 'raising_method', {})
