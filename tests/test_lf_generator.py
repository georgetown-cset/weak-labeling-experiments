import unittest
from unittest.mock import Mock
from context import weak_labeling
from snorkel.labeling import LabelingFunction

class LfGeneratorKeywordLookupTest(unittest.TestCase):
    def setUp(self):
        self.RELEVANT = 1
        self.IRRELEVANT = 0
        self.ABSTAIN = -1
        self.LFG = weak_labeling.lf_generator.LFGenerator()

        # the paper title and abstract could be anything
        self.paper_title = "front vehicle detection in video images based on temporal and spatial characteristics"
        self.paper_abstract = "Assisted driving and unmanned driving have been areas of focus for both industry and academia. Front-vehicle detection technology, a key component of both types of driving, has also attracted great interest from researchers. In this paper, to achieve front-vehicle detection in unmanned or assisted driving, a vision-based, efficient, and fast front-vehicle detection method based on the spatial and temporal characteristics of the front vehicle is proposed. First, a method to extract the motion vector of the front vehicle is put forward based on Oriented FAST and Rotated BRIEF (ORB) and the spatial position constraint. Then, by analyzing the differences between the motion vectors of the vehicle and those of the background, feature points of the vehicle are extracted. Finally, a feature-point clustering method based on a combination of temporal and spatial characteristics are applied to realize front-vehicle detection. The effectiveness of the proposed algorithm is verified using a large number of videos."
        return super().setUp()

    def test_data_import(self):
        file_paths = Mock()
        file_paths.configure_mock(wipo_data = "path/to/wipo_training_data.csv", keywords="path/to/keywords.csv")
        self.assertRaises(FileNotFoundError, self.LFG.data_import, file_paths.wipo_data, file_paths.keywords)

    def test_keyword_lookup(self):
        data_point, keywords = Mock(), Mock()
        data_point.configure_mock(papertitle=f"{self.paper_title}", abstract=f"{self.paper_abstract}")
        keywords.configure_mock(word_list=["vehicle detection", "assisted driving"])
        self.assertEqual(self.LFG.keyword_lookup(data_point, keywords.word_list, self.RELEVANT), 1)

        # it should raise a type error if number of arguments is insufficient
        self.assertRaises(TypeError, self.LFG.keyword_lookup, data_point, keywords.word_list)

        # it should abstain if the keywords are not found in the abstract
        self.assertEqual(self.LFG.keyword_lookup(data_point, ["general kenobi", "obi wan"], self.IRRELEVANT), -1)

    def test_make_keyword_lf(self):
        keywords = Mock()
        keywords.configure_mock(word_list=["vehicle detection"])
        lf = self.LFG.make_keyword_lf(keywords.word_list, label=self.RELEVANT)
        self.assertIsInstance(lf, LabelingFunction)
    
    def test_create_wordlist_set(self):
        phrases = Mock()
        phrases.configure_mock(phrase_list=["driver detection, vehicle identification", "driver recognition, driver detection"],
                                phrases_set={"driver detection", "vehicle identification", "driver recognition"})
        word_list = self.LFG.create_wordlist_set(phrases.phrase_list)
        self.assertEqual(word_list, phrases.phrases_set)

if __name__ == "__main__":
    unittest.main()
