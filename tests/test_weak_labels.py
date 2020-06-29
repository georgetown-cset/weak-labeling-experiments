import unittest
from unittest.mock import Mock
from context import weak_labeling


class KeywordLookupTest(unittest.TestCase):
    def setUp(self):
        self.RELEVANT = 1
        self.IRRELEVANT = 0
        self.ABSTAIN = -1
        return super().setUp()

    def test_keyword_lookup(self):
        # create a mock data point
        data_point = Mock()
        data_point.configure_mock(papertitle="this is the paper title", abstract="custom abstract with a few words")
        # it should return relevant if keywords match
        self.assertEqual(weak_labeling.weak_labels.keyword_lookup(data_point, ["custom", "few"], self.RELEVANT), 1)

        # it should return abstain if keywords are not found in data point
        self.assertEqual(weak_labeling.weak_labels.keyword_lookup(data_point, ["driver", "vehicle"], self.IRRELEVANT), -1)

        # it should return irrelevant if the keywords match but the label is irrelevant
        self.assertEqual(weak_labeling.weak_labels.keyword_lookup(data_point, ["this", "abstract"], 0), 0)


if __name__ == "__main__":
    unittest.main()
