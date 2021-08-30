import unittest
from pathlib import Path

import rationai.utils.dirstructure as tst


class TestDirstructure(unittest.TestCase):
    def setUp(self):
        self.dir_struct = tst.DirStructure()
        self.key = 'key'
        self.path = Path('some/random/path')

    def test_add_new_key(self):
        result = self.dir_struct.add(self.key, self.path)

        self.assertEqual(self.path, result)
        self.assertEqual(self.path, self.dir_struct.get(self.key))

    def test_add_existing_key(self):
        self.dir_struct.add(self.key, self.path)

        with self.assertRaises(ValueError):
            self.dir_struct.add(self.key, Path('some/other/path'))

        with self.assertRaises(ValueError):
            self.dir_struct.add(self.key, self.path)

    def test_get_non_existing_key(self):
        result = self.dir_struct.get(self.key)

        self.assertIsNone(result)

    def test_get_existing_key(self):
        self.dir_struct.add(self.key, self.path)

        self.assertEqual(self.path, self.dir_struct.get(self.key))
