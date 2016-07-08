#! encoding = utf-8
import numpy as np

def res_val_pair(result_list, val_list):
    ''' Generate dictionary pair a_dict[res]=val '''
    a_dict = {}
    # check length match
    if len(result_list) != len(val_list):
        RaiseError(IndexError)
    else:
        for i in range(len(result_list)):
            a_dict[result_list[i]] = val_list[i]

    return a_dict


def validate_data(file1, file2, tor=1e-6):
    ''' Validate data in file1 and file2.
        Return status_code: 0 -- identical
                            1 -- values agree within tolerance
                            2 -- disagree
    '''
    # load data
    data1 = np.loadtxt(file1, delimiter=',', skiprows=1)
    data2 = np.loadtxt(file2, delimiter=',', skiprows=1)

    if np.shape(data1) != np.shape(data2):
        return 2

    if np.all(data1 == data2):
        return 0
    else:
        diff = data1 - data2
        if np.average(np.power(diff, 2)) < tor:
            return 1
        else:
            return 2


def print_status(status_code, file1, file2):
    ''' Print out status_code '''

    status_msg = {0: 'Identical:',
                  1: 'Agree within tol 1e-6:',
                  2: 'Disagree:'}

    print('{:<22s} File {:s} and {:s}'.format(status_msg[status_code], file1, file2))

    return None


if __name__ == '__main__':

    # initialize status_code
    overall_status_code = 0
    overall_status_msg = {0: 'Validation Passed !',
                          1: 'Validation Acceptable ... Data agree within tolerance.',
                          2: 'Validation Failed ... Disagreements found in test data'}

    print('*'*20)
    print('Validating ...')

    # sample result file list
    result_list = ['v1result_single_bg5_mode1.csv',
                   'v1result_single_bg5_mode1_nobase.csv',
                   'v1result_single_bg5_mode1_spline.csv',
                   'v1result_single_bg5_mode2.csv',
                   'v1result_fullband_bg5_mode1.csv',
                   'v1result_fullband_bg5_mode1_nobase.csv',
                   'v1result_fullband_bg5_mode1_spline.csv'
                  ]

    # validation result file list
    val_list = ['val_single_bg5_mode1.csv',
                'val_single_bg5_mode1_nobase.csv',
                'val_single_bg5_mode1_spline.csv',
                'val_single_bg5_mode2.csv',
                'val_fb_bg5_mode1.csv',
                'val_fb_bg5_mode1_nobase.csv',
                'val_fb_bg5_mode1_spline.csv',
               ]

    pairs = res_val_pair(result_list, val_list)
    for res, val in pairs.items():
        status_code = validate_data(res, val)
        print_status(status_code, res, val)
        overall_status_code = max(overall_status_code, status_code)

    print('*'*20)
    print(overall_status_msg[overall_status_code])
