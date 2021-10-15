import unittest

from hypercrl.tools import Hparams


class TestDefaultArg(unittest.TestCase):
    """
    Unittests for the robotic priors
    """

    def setUp(self) -> None:
        self.dimension = 128
        self.batch_size = 8

        hparams = Hparams()
        hparams.test = "Nice"
        hparams.list = ["Nice"] * 2
        hparams.none = None

        hparams.hparams = Hparams()
        hparams.hparams.test = "Nice"
        hparams.hparams.list = ["Nice"] * 2
        hparams.hparams.none = None

        hparams.hparams.hparams = Hparams()
        hparams.hparams.hparams.test = "Nice"
        hparams.hparams.hparams.list = ["Nice"] * 2
        hparams.hparams.hparams.none = None

        self.hparams = hparams

    def test_hparams_to_dict(self) -> None:
        result = self.hparams.to_dict()
        self.assertEqual(str(result),
                         "{'test': 'Nice', 'list': \"['Nice', 'Nice']\", 'none': 'None', 'hparams.test': 'Nice', 'hparams.list': \"['Nice', 'Nice']\", 'hparams.none': 'None', 'hparams.hparams.test': 'Nice', 'hparams.hparams.list': \"['Nice', 'Nice']\", 'hparams.hparams.none': 'None'}")

    def test_hparams_to_filename(self) -> None:
        result = self.hparams.to_filename()
        self.assertEqual(result,
                         'test:Nice+list:[\'Nice\',\'Nice\']+none:None+hparams.test:Nice+hparams.list:[\'Nice\',\'Nice\']+hparams.none:None+hparams.hparams.test:Nice+hparams.hparams.list:[\'Nice\',\'Nice\']+hparams.hparams.none:None')
