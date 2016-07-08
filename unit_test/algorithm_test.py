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


if __name__ == '__main__':
    unittest.main()
