import unittest
from unittest.mock import Mock
from context import weak_labeling

class ClassBalanceEstimatorTest(unittest.TestCase):
    def setUp(self):

        return super().setUp()

    def test_format_keyword(self):
        kw = Mock()
        kw.configure_mock(word="vehicle detection", text="sample vehicle detection text for test")
        self.assertRegex(kw.text, weak_labeling.class_balance_estimator.format_keyword(kw.word))

if __name__ == "__main__":
    unittest.main()