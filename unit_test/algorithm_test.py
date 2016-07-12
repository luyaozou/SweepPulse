#! encoding = utf-8

''' Algorithm unit tests for SweepPulse.py '''

import sweep as sp
import numpy as np
import argparse
import unittest


TOL = 1e-6      # float number tolerance


def parser_gen(args_list):
    ''' Simulate an input parser '''

    parser = argparse.ArgumentParser(description=__doc__,
                                    epilog='--- Luyao Zou, July 2016 ---')
    parser.add_argument('inten', nargs=1, help='Intensity data file')
    parser.add_argument('-bg', nargs=1, type=int,
                        help='''The ordinal number of the full sweep to use
                                as background. If not specified,
                                assume no background subtraction is required,
                                and all odd sweep are averaged together. ''')
    parser.add_argument('-cf', nargs=1,
                        help='''Single center frequency (MHz) or a file listing
                                several center frequencies. If neither
                                specified, set at 0, and assume intensity is
                                a single sweep scan.''')
    parser.add_argument('-bdwth', nargs=1, type=float,
                        help='''Full frequency sweep band width (MHz).
                                If not specified while freq file is available,
                                get sweep window from the difference of the
                                first two data points in the freq file,
                                assuming frequency data points are evenly spaced
                                and matches the band width. Default is 1.''')
    parser.add_argument('-box', nargs=1, type=int,
                        help='Boxcar smooth window. Must be an odd integer.')
    parser.add_argument('-lo', nargs=1,
                        help='''LO file. If not specified, command line
                                interactive questions will be invoked.''')
    parser.add_argument('-o', nargs=1,
                        help='''Output file name. If not specified,
                                default name will be used.''')
    parser.add_argument('-delay', nargs=1, type=int,
                        help=''' Delay of detector response by number of
                                 data points. Default is 0.''')
    parser.add_argument('-spline', action='store_true',
                        help='Fit spline to subtract baseline. Optional')
    parser.add_argument('-nobase', action='store_true',
                        help='Disable ALL baseline removal functionality.')

    return parser.parse_args(args_list)


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
        test_x_out = sp.box_car(self.box_x_in, 1)
        test_y_out = sp.box_car(self.box_y_in, 1)
        self.assertEqual(self.box_x_in.shape, test_x_out.shape)
        self.assertEqual(self.box_y_in.shape, test_y_out.shape)
        self.assertTrue(np.linalg.norm(self.box_x_in - test_x_out) < TOL)
        self.assertTrue(np.linalg.norm(self.box_y_in - test_y_out) < TOL)
        # test box-car win=3 -- real case

        test_x_out = sp.box_car(self.box_x_in, 3)
        test_y_out = sp.box_car(self.box_y_in, 3)
        self.assertEqual(self.box_x_out.shape, test_x_out.shape)
        self.assertEqual(self.box_y_out.shape, test_y_out.shape)
        self.assertTrue(np.linalg.norm(self.box_x_out - test_x_out) < TOL)
        self.assertTrue(np.linalg.norm(self.box_y_out - test_y_out) < TOL)

    def test_box_car_array(self):

        print('\nTest box-car algorithm broadcast to 2D arrays')

        array_x_in = np.tile(self.box_x_in, (10, 1)).transpose()
        array_x_out = np.tile(self.box_x_out, (10, 1)).transpose()
        test_x_out = np.apply_along_axis(sp.box_car, 0, array_x_in, 3)
        self.assertEqual(array_x_out.shape, test_x_out.shape)
        self.assertTrue(np.linalg.norm(array_x_out - test_x_out) < TOL)


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


class ArrayShapes(unittest.TestCase):
    ''' Test array manuplication and reshape routines. '''

    x_1d = np.arange(100)
    y_1d = np.sin(x_1d)
    x_2d = np.arange(1000).reshape((100, 10), order='F')
    y_2d = np.radians(x_2d)

    def test_delay_inten(self):

        print('\nTest delay intensity function')

        inten_2d = np.tile(self.x_1d, (100, 1)).transpose()
        delay = 5
        freq_2d = 1

        goal_1d = np.append(np.arange(100-delay) + delay, np.arange(delay))
        goal_2d = np.tile(goal_1d, (100, 1)).transpose()

        test_1d = sp.delay_inten(self.x_1d, delay)
        test_2d = sp.delay_inten(inten_2d, delay)

        print('\nTest delay intensity 1D array')
        self.assertTrue(np.linalg.norm(goal_1d - test_1d) < TOL)
        print('\nTest delay intensity 2D array')
        self.assertTrue(np.linalg.norm(goal_2d - test_2d) < TOL)

    def test_proc_wb(self):

        print('\nTest broad band process function')

        args_list = ['NONE.dat', '-nobase']
        print('\nInput parameters: ' + ' '.join(args_list))
        goal_y = self.y_1d
        test_y = sp.proc_wb(self.x_1d, self.y_1d, parser_gen(args_list))
        self.assertEqual(self.x_1d.shape, test_y.shape)
        self.assertTrue(np.linalg.norm(goal_y - test_y) < TOL)

        args_list = ['NONE.dat']
        print('\nInput parameters: ' + ' '.join(args_list))
        test_y = sp.proc_wb(self.x_1d, self.y_1d, parser_gen(args_list))
        self.assertEqual(self.x_1d.shape, test_y.shape)

        args_list = ['NONE.dat', '-spline']
        print('\nInput parameters: ' + ' '.join(args_list))
        test_y = sp.proc_wb(self.x_1d, self.y_1d, parser_gen(args_list))
        self.assertEqual(self.x_1d.shape, test_y.shape)

    def test_proc_nb(self):

        print('\nTest narrow band process function')
        pass

    def test_flat_wave(self):

        print('\nTest flat_wave function')
        # test 1D data, should just return the original
        x_flat, y_flat = sp.flat_wave(self.x_1d, self.y_1d)
        self.assertEqual(self.x_1d.shape, x_flat.shape)
        self.assertEqual(self.y_1d.shape, y_flat.shape)
        self.assertTrue(np.linalg.norm(self.x_1d - x_flat) < TOL)
        self.assertTrue(np.linalg.norm(self.y_1d - y_flat) < TOL)

        # test 2D data
        x_goal = self.x_2d.flatten('F')
        y_goal = self.y_2d.flatten('F')
        x_flat, y_flat = sp.flat_wave(self.x_2d, self.y_2d)
        self.assertEqual(x_goal.shape, x_flat.shape)
        self.assertEqual(y_goal.shape, y_flat.shape)
        self.assertTrue(np.linalg.norm(x_goal - x_flat) < TOL)
        #self.assertTrue(np.linalg.norm(y_goal - y_flat) < TOL)




class DbFuncs(unittest.TestCase):
    ''' Test debaseline algorithms '''

    def test_glue_inten(self):

        print('\nTest glue intensity algorithm')

        pass


if __name__ == '__main__':
    unittest.main()
