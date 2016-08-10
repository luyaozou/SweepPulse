#! encoding = utf-8

''' Function/file handle unit tests for SweepPulse.py '''

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
    parser.add_argument('-fg', nargs=1, type=int,
                        help='''The ordinal number of the signal sweep.
                                Default is 1. ''')
    parser.add_argument('-bg', nargs=1, type=int,
                        help='''The ordinal number of the background sweep. If
                                bg == fg, simply extract the fg sweep without
                                background subtraction. If not specified, all
                                odd sweep are averaged together. ''')
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


class ErrMsgPrint(unittest.TestCase):
    ''' Test error message printing '''

    def test_err_msg_print(self):
        print('\nTest error message printing')

        self.assertEqual(sp.err_msg_str('f', 0), '')
        self.assertEqual(sp.err_msg_str('f', 1), 'f does not exist')
        self.assertEqual(sp.err_msg_str('f', 2), 'f format is not supported')
        self.assertEqual(sp.err_msg_str('f', 3), 'f contains an object array that is not allowed to load')


class LoadFile(unittest.TestCase):
    ''' Test file load system '''

    normal_txt_file = 'validation_test/sample_input_single_inten.dat'
    normal_npy_file = 'validation_test/sample_input_fb_wavybase-noline_lo.npy'
    mal_format_file = 'validation_test/run_val_WIN.bat'
    non_exist_file = 'Nowhere.csv'
    y_single = 'validation_test/sample_input_single_inten.dat'
    x_fb = 'validation_test/sample_input_fb_numerous-lines_freq.dat'
    y_fb = 'validation_test/sample_input_fb_numerous-lines_inten.npy'

    def test_single_file_txt_fmt(self):
        print('\nTest single file loader')
        loaded = sp.load_single_file(self.normal_txt_file)
        self.assertTrue(isinstance(loaded, np.ndarray))
        self.assertEqual(loaded.shape, (5000,))

        loaded = sp.load_single_file(self.mal_format_file)
        self.assertTrue(isinstance(loaded, type(None)))
        loaded = sp.load_single_file(self.non_exist_file)
        self.assertTrue(isinstance(loaded, type(None)))

    def test_loader_single_y(self):
        print('\nTest data loader -- single y file')
        args_list = [self.y_single, '-lo', self.normal_txt_file]
        x, y = sp.load_data(parser_gen(args_list))
        self.assertEqual(x.shape, (1000,))
        self.assertEqual(y.shape, (1000,))

    def test_loader_single_npy(self):
        print('\nTest data loader -- single npy binary file')
        loaded = sp.load_single_file(self.normal_npy_file)
        self.assertTrue(isinstance(loaded, np.ndarray))
        self.assertEqual(loaded.shape, (2500, 55))

    def test_loader_fb_xy(self):
        print('\nTest data loader -- full band x, y file')
        args_list = [self.y_fb, '-cf', self.x_fb, '-lo', self.normal_txt_file,
                     '-bdwth', '18']
        print('\nInput parameters: ' + ' '.join(args_list))
        x, y = sp.load_data(parser_gen(args_list))
        self.assertEqual(x.shape, (1000, 57))
        self.assertEqual(y.shape, (1000, 57))

        args_list = [self.y_fb, '-cf', self.x_fb, '-lo', self.normal_txt_file,
                     '-bdwth', '18', '-delay', '4', '-bg', '5']
        print('\nInput parameters: ' + ' '.join(args_list))
        x, y = sp.load_data(parser_gen(args_list))
        self.assertEqual(x.shape, (996, 57))
        self.assertEqual(y.shape, (996, 57))

    def test_analyze_txt_fmt(self):
        print('\nTest file format analyzer')

        print('\nTest valid single array without header')
        delm, hd, eof = sp.analyze_txt_fmt(self.x_fb)
        self.assertEqual(delm, ',')
        self.assertEqual(hd, 0)

        print('\nTest valid comma delimited text without header')
        delm, hd, eof = sp.analyze_txt_fmt(self.y_fb)
        self.assertEqual(delm, ',')
        self.assertEqual(hd, 0)

        print('\nTest valid tab delimited text with 1 header row')
        delm, hd, eof = sp.analyze_txt_fmt('unit_test/sample_tab_delm_1_header.txt')
        self.assertEqual(delm, '\t')
        self.assertEqual(hd, 1)

        print('\nTest valid white space delimited text with 2 header rows')
        delm, hd, eof = sp.analyze_txt_fmt('unit_test/sample_space_delm_2_header.txt')
        self.assertEqual(delm, ' ')
        self.assertEqual(hd, 2)

        print('\nTest invalid text')
        delm, hd, eof = sp.analyze_txt_fmt(self.mal_format_file)
        self.assertTrue(isinstance(delm, type(None)))
        self.assertEqual(hd, 16)


if __name__ == '__main__':
    unittest.main()
