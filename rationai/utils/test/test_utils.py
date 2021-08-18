import unittest

from rationai.utils import utils as tst


class CallableHasSignature(unittest.TestCase):
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
