#! encoding = utf-8

''' Function/file handle unit tests for SweepPulse.py '''

import sweep as sp
import numpy as np
import unittest

TOL = 1e-6      # float number tolerance


class ErrMsgPrint(unittest.TestCase):
    ''' Test error message printing '''

    def test_err_msg_print(self):

        self.assertEqual(sp.err_msg_str('f', 0), '')
        self.assertEqual(sp.err_msg_str('f', 1), 'f does not exist')
        self.assertEqual(sp.err_msg_str('f', 2), 'f format is not supported')


if __name__ == '__main__':
    unittest.main()
