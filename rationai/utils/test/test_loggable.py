import unittest
import io
from contextlib import redirect_stdout


from rationai.utils import loggable as tst


class TestLoggable(unittest.TestCase):
    def test_can_set_path_to_log(self):
        self.assertIsNone(tst.Loggable.PATH_TO_LOG)

        tst.Loggable.set_log_path('/some/path')

        self.assertEqual('/some/path', tst.Loggable.PATH_TO_LOG)

        tst.Loggable.set_log_path(None)
        self.assertIsNone(tst.Loggable.PATH_TO_LOG)

    def test_logs_proper_message_and_value_to_stdout(self):

        @tst.Loggable('test_key')
        def test_sum(a, b):
            return a + b

        # redirect stdout to a StringIO object so that we can capture it
        with io.StringIO() as buffer, redirect_stdout(buffer):
            test_sum(3, 5)
            logged_output = buffer.getvalue()

        self.assertEqual('test_key: 8\n', logged_output)
