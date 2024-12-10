import unittest
import rhythm_regression.unit_conversion as uc

class TestUnitConversion(unittest.TestCase):

    def test_hz_to_ms_per_note(self):
        self.assertAlmostEqual(uc.hz_to_ms_per_note(12), 83.333, places=3)

    def test_hz_to_8rbpm(self):
        self.assertEqual(uc.hz_to_8rbpm(12), 180)

    def test_hz_to_12rbpm(self):
        self.assertEqual(uc.hz_to_12rbpm(12), 120)

    def test_hz_to_16rbpm(self):
        self.assertEqual(uc.hz_to_16rbpm(12), 90)

    def test_ms_per_note_to_hz(self):
        self.assertAlmostEqual(uc.ms_per_note_to_hz(83.333), 12, places=3)

    def test_ms_per_note_to_8rbpm(self):
        self.assertAlmostEqual(uc.ms_per_note_to_8rbpm(83.333), 180, places=2)

    def test_ms_per_note_to_12rbpm(self):
        self.assertAlmostEqual(uc.ms_per_note_to_12rbpm(83.333), 120, places=3)

    def test_ms_per_note_to_16rbpm(self):
        self.assertAlmostEqual(uc.ms_per_note_to_16rbpm(83.333), 90, places=3)

    def test_8rbpm_to_hz(self):
        self.assertEqual(uc._8rbpm_to_hz(180), 12)

    def test_8rbpm_to_ms_per_note(self):
        self.assertAlmostEqual(uc._8rbpm_to_ms_per_note(180), 83.333, places=3)

    def test_8rbpm_to_12rbpm(self):
        self.assertEqual(uc._8rbpm_to_12rbpm(180), 120)

    def test_8rbpm_to_16rbpm(self):
        self.assertEqual(uc._8rbpm_to_16rbpm(180), 90)

    def test_12rbpm_to_hz(self):
        self.assertEqual(uc._12rbpm_to_hz(120), 12)

    def test_12rbpm_to_ms_per_note(self):
        self.assertAlmostEqual(uc._12rbpm_to_ms_per_note(120), 83.333, places=3)

    def test_12rbpm_to_8rbpm(self):
        self.assertEqual(uc._12rbpm_to_8rbpm(120), 180)

    def test_12rbpm_to_16rbpm(self):
        self.assertEqual(uc._12rbpm_to_16rbpm(120), 90)

    def test_16rbpm_to_hz(self):
        self.assertEqual(uc._16rbpm_to_hz(90), 12)

    def test_16rbpm_to_ms_per_note(self):
        self.assertAlmostEqual(uc._16rbpm_to_ms_per_note(90), 83.333, places=3)

    def test_16rbpm_to_8rbpm(self):
        self.assertEqual(uc._16rbpm_to_8rbpm(90), 180)

    def test_16rbpm_to_12rbpm(self):
        self.assertEqual(uc._16rbpm_to_12rbpm(90), 120)


if __name__ == '__main__':
    unittest.main()