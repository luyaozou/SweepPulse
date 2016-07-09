#! encoding = utf-8

''' Algorithm unit tests for SweepPulse.py '''

import sweep as sp
import numpy as np
import unittest

TOL = 1e-6      # float number tolerance

class BoxcarSmooth(unittest.TestCase):
    ''' Test box-car smooth function '''

    # know input:output box-car window pair
    box_win_pair = {1:1, 2:3, 3:3, 0:1, -1:1, -2:3}
    # know input output box-car x,y waveform
    box_x_in = np.arange(10)
    box_x_out = np.arange(8) + 1
    box_y_in = np.arange(20)
    box_y_out = np.arange(18) + 1

    def test_box_window(self):
        print('\nTest box-car window generator')
        for pair_in, pair_out in self.box_win_pair.items():
            test = sp.box_win(pair_in)
            self.assertEqual(pair_out, test)

    def test_box_car(self):
        print('\nTest box-car algorithm output')

        # test box-car win=1 -- output original
        test_x_out, test_y_out = sp.box_car(self.box_x_in, self.box_y_in, 1)
        self.assertEqual(self.box_x_in.shape, test_x_out.shape)
        self.assertEqual(self.box_y_in.shape, test_y_out.shape)
        self.assertTrue(np.linalg.norm(self.box_x_in - test_x_out) < TOL)
        self.assertTrue(np.linalg.norm(self.box_y_in - test_y_out) < TOL)
        # test box-car win=3 -- real case

        test_x_out, test_y_out = sp.box_car(self.box_x_in, self.box_y_in, 3)
        self.assertEqual(self.box_x_out.shape, test_x_out.shape)
        self.assertEqual(self.box_y_out.shape, test_y_out.shape)
        self.assertTrue(np.linalg.norm(self.box_x_out - test_x_out) < TOL)
        self.assertTrue(np.linalg.norm(self.box_y_out - test_y_out) < TOL)


class DataLoader(unittest.TestCase):
    ''' Test data loader functionality '''

    cf_single_zero = 0
    cf_single_nonzero =  100000.
    bdwth = 12.
    cf_array = np.arange(20)*bdwth + 100000.
    pts = 500
    band = np.arange(pts)/pts - 0.5

    def test_freq_reconstructor(self):
        print('\nTest frequency reconstructor')

        # case 1, cf=0, known bdwth, sweep up
        goal = self.band * self.bdwth
        test = sp.reconstr_freq(self.cf_single_zero, self.pts, sweep_up=True, bdwth=self.bdwth)
        self.assertEqual(goal.shape, test.shape)
        self.assertTrue(np.linalg.norm(goal - test) < TOL)

        # case 2, cf=100GHz, known bdwth, sweep down
        goal = self.cf_single_nonzero - self.band*self.bdwth
        test = sp.reconstr_freq(self.cf_single_nonzero, self.pts, sweep_up=False, bdwth=self.bdwth)
        self.assertEqual(goal.shape, test.shape)
        self.assertTrue(np.linalg.norm(goal - test) < TOL)

        # case 3, cf=100GHz, unknown bdwth, sweep up
        goal = self.band + self.cf_single_nonzero
        test = sp.reconstr_freq(self.cf_single_nonzero, self.pts, sweep_up=True)
        self.assertEqual(goal.shape, test.shape)
        self.assertTrue(np.linalg.norm(goal - test) < TOL)

        # case 4, cf=array, known bandwidth, sweep down
        goal = -np.arange(20*self.pts)/(20*self.pts)*240 + 100234.
        goal = np.fliplr(goal.reshape((self.pts, 20), order='F'))
        test = sp.reconstr_freq(self.cf_array, self.pts, bdwth=self.bdwth, sweep_up=False)
        self.assertEqual(goal.shape, test.shape)
        self.assertTrue(np.linalg.norm(goal - test) < TOL)

    def test_load_data(self):
        print('\nTest data loader')

        pass


if __name__ == '__main__':
    unittest.main()
