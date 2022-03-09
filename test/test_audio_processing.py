import unittest
import rhythm_regression.audio_processing as ap

class TestUnitConversion(unittest.TestCase):

    def test_amplitude_envelope(self):

        signal = ap.np.array([-5,3,6,4,4,-2,-1,-7,-1,0,2,5,9,10,7,1,-3,-4,-5])

        samples, ae = ap.amplitude_envolope(signal, frame_size=5, hop_length=5)
        self.assertEqual(samples.size, ae.size)
        self.assertTrue((samples == ap.np.array(range(0, len(signal), 5))).all())
        self.assertTrue((ae == ap.np.array([6,0,10,1])).all())

        samples, ae = ap.amplitude_envolope(signal, frame_size=5, hop_length=3)
        self.assertEqual(samples.size, ae.size)
        self.assertTrue((samples == ap.np.array(range(0, len(signal), 3))).all())
        self.assertTrue((ae == ap.np.array([6,4,2,10,10,1,-5])).all())

        samples, ae = ap.amplitude_envolope(signal, frame_size=1, hop_length=1)
        self.assertEqual(samples.size, ae.size)
        self.assertTrue((samples == ap.np.array(range(0, len(signal), 1))).all())
        self.assertTrue((ae == signal).all())

        samples, ae = ap.amplitude_envolope(signal, frame_size=2, hop_length=5)
        self.assertEqual(samples.size, ae.size)
        self.assertTrue((samples == ap.np.array(range(0, len(signal), 5))).all())
        self.assertTrue((ae == ap.np.array([3,-1,5,1])).all())

    def test_rms_energy_transients(self):
        pass
        
        #signal = ap.np.array([-5,3,6,4,4,-2,-1,-7,-1,0,2,5,9,10,7,1,-3,-4,-5])

        #transients = ap.rms_energy_transients(signal, sampling_rate=1, frame_length=5, hop_length=3, amplitude_threshold=0)
        # rmse is array([[3.74165739, 4.02492236, 3.76828874, 3.97492138, 7.19722169, 5.91607978, 3.16227766]])
        #self.assertTrue((transients == ap.np.array([1, 4])).all())

        # test threshold

        # test frame_length and hop_length



    def test_arg_where_local_max(self):
        
        # test for up down pattern
        self.assertTrue((ap.arg_where_local_max([0,1,2,3,4,3,2,1,2,3,4,3,2,1]) == ap.np.array([4,10])).all())
        self.assertTrue((ap.arg_where_local_max(ap.np.array([-1,0,1,0,-1,0,-1,0,-1,0,1,0])) == ap.np.array([2,5,7,10])).all())

        # test for no local maxima
        self.assertTrue((ap.arg_where_local_max([5,6,7,8]) == ap.np.array([])).all())
        self.assertTrue((ap.arg_where_local_max(ap.np.array([-100,-50,0,50,100])) == ap.np.array([])).all())
        self.assertTrue((ap.arg_where_local_max([-1,-2,-3,-4,-5]) == ap.np.array([])).all())
        self.assertTrue((ap.arg_where_local_max(ap.np.array([9,8,7,6,5])) == ap.np.array([])).all())

        # test for equal on left, decrease on right
        self.assertTrue((ap.arg_where_local_max([5,5,4,4]) == ap.np.array([1])).all())
        self.assertTrue((ap.arg_where_local_max(ap.np.array([3,3,2,2,1,1,0,0,-1])) == ap.np.array([1,3,5,7])).all())

        # test empty array
        self.assertTrue((ap.arg_where_local_max([]) == ap.np.array([])).all())
        self.assertTrue((ap.arg_where_local_max(ap.np.array([])) == ap.np.array([])).all())


if __name__ == '__main__':
    unittest.main()