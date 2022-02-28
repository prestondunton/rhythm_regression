import unittest
import rhythm_regression.audio_processing as ap

import os

class TestUnitConversion(unittest.TestCase):

    def setUp(self):
        
        test_dir = os.path.dirname(__file__)
        sample_path = os.path.join(test_dir, 'test_data', 'Sample 1.m4a')
        self.sample, _ = ap.librosa.load(sample_path)

    def test_amplitude_envelope(self):

        default_samples, default_ae = ap.amplitude_envolope(self.sample)
        self.assertEqual(len(default_samples), len(default_ae))
        self.assertTrue((default_samples == ap.np.array(range(0, len(self.sample), 512))).all())

        first_transient = (20000 <= default_samples) & (default_samples < 30000)
        self.assertEqual(max(default_ae[first_transient]), max(self.sample[24000:24500]))

        self.assertEqual(ap.np.argwhere(default_ae == 0).flatten().tolist()[-1], len(default_ae) - 1)
        self.assertEqual(ap.np.argwhere(default_ae < 0).size, 0)

        self.assertEqual(self.sample.size - max(default_samples), self.sample.size % 512)
        
        


if __name__ == '__main__':
    unittest.main()