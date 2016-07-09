# encoding = utf8
''' This script processes the linear-sweep data for pulsed experiment.
It is REQUIRED that the data takes full cycles of sweeps. '''

import argparse
import numpy as np
from scipy import interpolate
from scipy.optimize import leastsq

# ------------------------------------------
# ------ MESSAGE CONSTANT DECLARATION ------
# ------------------------------------------
FILE_ERR_MSG = {0: '',                              # Silent
                1: '{:s} does not exist',           # FileNotFoundError
                2: '{:s} format is not supported',  # Format Issue
                }

# ----------------------------------
# ------ Function Declaration ------
# ----------------------------------

def box_smooth(a, box_win):
    ''' Boxcar smooth routine. Accepts only vector '''
    return np.convolve(a, np.ones(box_win), 'valid')/box_win

def glue_sweep(y):
    ''' shift each sweep intensity to make the end of the previous sweep is
    equal to the start of the next sweep, so that the spectrum is
    continuous. Accepts matrix '''
    # Get the difference of the end of col and the start of col+1
    col_shift = np.roll(y[-1, :], 1) - y[0, :]
    col_shift[0] = 0
    col_shift_accum = np.cumsum(col_shift)
    # Apply shift correction to all columns
    return y + np.tile(col_shift_accum, (y.shape[0], 1))

def db_spline(y):
    ''' Background subtraction by fitting spline to the baseline.
    Accepts only vector '''
    # Because of the change of refraction index, the background
    # does not exactly match the signal. This creates a curved baseline
    # after background subtraction. Try to fit a B-spline to the
    # baseline.
    x = np.arange(len(y))
    # Construct weight
    weight = np.exp(-np.power(y - y.max(), 2))
    # Interpolate spline
    spline = interpolate.UnivariateSpline(x, y, w=weight, k=5)
    return y - spline(x)

def db_diff(freq, inten):
    ''' Derivative. Accepts only vector '''
    inten_diff = np.diff(inten)
    freq_diff = freq[:-1] + np.diff(freq)/2
    return freq_diff, inten_diff

def db_poly(y, deg):
    ''' Polynomial baseline clean. Accepts on vector '''
    x = np.arange(len(y))
    popt = np.polyfit(x, y, deg)
    return y - np.polyval(popt, x)

def sinusoidal_res(p, x, y):
    ''' Residual function of a sinusoidal wave to match the baseline
    wiggle. This includes a short-period sine wave to correct the
    detector response, and a long-period sine wave to correct the
    overall wavy structure after flatten the sweeps. '''
    # y^ = a*sin(2pi*x/T + phi) + m*x + b
    a, t, phi = p
    res = y - a*np.sin(2*np.pi*x/t + phi)
    return res

def sinusoidal_eval(x, p):
    ''' Evaluate the sinusoidal wave '''
    return p[0]*np.sin(2*np.pi*x/p[1] + p[2])

def db_sinusoidal(y, init):
    ''' Apply linear shift correction on a sweep. Accepts only vector '''
    x = np.arange(len(y))
    plsq, pcoval = leastsq(sinusoidal_res, init, args=(x, y))
    shift = sinusoidal_eval(x, plsq)
    return y - shift

def extract_sig_bg(freq, inten, bg, sweep, delay, single_freq):
    ''' Seperate signal waveforms from background waveforms '''
    # Check if the backgroud is the last sweep
    bg_last_sweep = (bg == sweep)

    if bg_last_sweep and delay:
        if single_freq:
            # If the background is the last sweep,  we have to drop the last
            # few points in case of any detector delay
            inten_signal = inten[delay:sweep_range]
            inten_bg = inten[(bg-1)*sweep_range+delay:]
            freq = freq[:-delay]
        else:
            inten_signal = inten[delay:sweep_range, :]
            inten_bg = inten[(bg-1)*sweep_range+delay:, :]
            freq = freq[:-delay, :]
    # If the background is not the last sweep, we can use the full sweep
    elif delay:
        if single_freq:
            inten_signal = inten[delay:sweep_range+delay]
            inten_bg = inten[(bg-1)*sweep_range+delay:bg*sweep_range+delay]
        else:
            inten_signal = inten[delay:sweep_range+delay, :]
            inten_bg = inten[(bg-1)*sweep_range+delay:bg*sweep_range+delay, :]
    # If there is no delay
    elif single_freq:
        inten_signal = inten[:sweep_range]
        inten_bg = inten[(bg-1)*sweep_range:bg*sweep_range]
    else:
        inten_signal = inten[:sweep_range, :]
        inten_bg = inten[(bg-1)*sweep_range:bg*sweep_range, :]

    # Check if background is at an odd-number sweep or an even-number sweep.
    # If even, the sequency of the intensity needs to be flipped to match
    # the 1st sweep.
    if not (bg % 2):  # even-number background
        inten_bg = np.flipud(inten_bg)
    else:
        pass

    return freq, inten_signal, inten_bg


#--------- start refactoring here ----------
def box_car(x, y, win):
    ''' Boxcar smooth.

    Arguments:
    x   -- x frequency array, np.array
    y   -- y intensity array, np.array
    win -- box-car window, integer

    Returns:
    x_new -- new x array, np.array
    y_new -- new y array, np.array
    '''

    if win == 1:
        return x, y
    else:
        x_new = x[win//2:-win//2+1]
        y_new = np.convolve(y, np.ones(win), 'valid')/win
        return x_new, y_new


def box_win(win):
    ''' Verify boxcar window.

    Arguments:
    win -- box-car window, integer

    Returns:
    win_verified -- verified box-car window, odd integer
    '''

    win_verified = abs(win)
    if win_verified == 0:
        return 1
    elif not (win_verified % 2):
        return win_verified + 1
    else:
        return win_verified


def db_wb(y, if_spline):
    ''' Wide band debaseline.

    Arguments:
    y -- y intensity data array, np.array
    if_spline -- spline fit option, boolean

    Returns:
    y_db, np.array
    '''

    y_db = db_poly(y, deg=1)
    if if_spline:
        y_db = db_spline(y_db)
    else:
        pass

    return y_db


def err_msg_str(f, err_code, msg=FILE_ERR_MSG):
    ''' Generate file error message string

    Arguments:
    f        -- file name, str
    err_code -- error code, int
    msg      -- error message, dict

    Returns:
    msg_str -- formated error message, str
    '''

    return (msg[err_code]).format(f)


def load_data(args):
    ''' Load all data files specified from input arguments.

    Arguments:
    args -- input argument, argparse Object

    Returen:
    freq -- frequency waveform, np.array 1D
    inten -- intensity waveform, np.array 1D/2D
    '''

    # load intensity file
    inten = load_single_file(args.inten[0])
    # exit if intensity file is not loaded correctly
    if not inten:
        exit()

    # load lo file if available
    if args.lo:
        lo = load_single_file(args.lo[0])
        sweep_num = np.count_nonzero(np.delete(lo*np.roll(lo, 1), 0) < 0)
        sweep_up = lo[0] < 0
    else:
        # no lo file, invoke interactive interface
        sweep_num, sweep_up = interactive()

    # number of points in a single sweep
    pts = inten.shape[0] / sweep_num

    # try bandwidth
    if args.bdwth:
        bdwth = args.bdwth[0]
    else:
        bdwth = 1.

    if args.cf:
        try:
            center_freq = float(args.cf[0])
        except ValueError:
            center_freq = load_single_file(args.cf[0])
            bdwth = center_freq[1] - center_freq[0]
    else:
        center_freq = 0

    freq = reconstr_freq(center_freq, pts, sweep_up=sweep_up, bdwth=bdwth)

    return freq, inten


def load_single_file(file_name):
    ''' Load single data file & raise exceptions.

    Arguments:
    file_name -- input file name, str

    Returns:
    data -- np.array
    '''

    try:
        data = np.loadtxt(file_name, delimiter=',')
        return data
    except FileNotFoundError:
        print(err_msg_str(file_name, 1))
        return None
    except ValueError:
        print(err_msg_str(file_name, 2))
        return None


def interactive():
    ''' Command line interactive interface for sweep information input.
    For mode 0, i.e. no LO data available only.

    Returns:
    sweep_num -- number of full sweeps, int
    sweep_up -- first sweep increases in frequency, boolean
    '''

    while True:     # Get number of sweeps from user input & Error handling
        try:
            typed = input('Input number of full sweeps: ').split()
            sweep_num = int(typed[0])
        except ValueError:
            typed = input('''Must be an integer! Retype: ''').split()

    # Ask if the first sweep goes up
    typed = input('Does the first sweep go up? Y|n ')
    sweep_up = typed in ('y', 'Y', 'yes', 'Yes', 'YES')

    return sweep_num, sweep_up


def reconstr_freq(center_freq, pts, sweep_up=True, bdwth=1.):
    ''' Reconstruct frequency array.

    Arguments:
    center_freq -- center frequency of each sweep. float or np.array
    pts -- dimension of the frequency array. int
    **sweep_up -- first sweep frequency increases. defautl True. boolean
    **bdwth -- sweep bandwidth (MHz), default 1. float

    Returns:
    freq -- frequency array, np.array 1D/2D
    '''

    if sweep_up:
        single_band = bdwth * (np.arange(pts)/pts - 0.5)
    else:
        single_band = bdwth * (0.5 - np.arange(pts)/pts)

    if isinstance(center_freq, np.ndarray):
        freq = np.tile(single_band, (len(center_freq), 1)).transpose() + \
               np.tile(center_freq, (pts, 1))
    else:
        freq = single_band + center_freq

    return freq


def save_output(out_name, out_data):
    ''' Output data in csv format.

    Arguments:
    out_name -- output data file name, str
    out_data -- output xy data, np.array

    Returns: None
    '''

    np.savetxt(out_name, out_data, header='freq,inten', delimiter=',',
               newline='\n', fmt='%.10g', comments='')
    print('-- {:s} Saved --'.format(out_name))

    return None


def arg():
    ''' Input arguments parser. Returns: argparse Object.'''

    parser = argparse.ArgumentParser(description=__doc__,
                                    epilog='--- Luyao Zou, April 2015 ---')
    parser.add_argument('inten', nargs=1, help='Intensity data file')
    parser.add_argument('-bg', nargs=1, type=int,
                        help='''The ordinal number of the full sweep to use
                                as background. If not specified,
                                assume no background subtraction is required,
                                and all sweep cycles are averaged together. ''')
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
    parser.add_argument('-spline', action='store_true',
                        help='Fit spline to subtract baseline. Optional')
    parser.add_argument('-nobase', action='store_true',
                        help='Disable ALL baseline removal functionality.')

    return parser.parse_args()


if __name__ == '__main__':

    input_args = arg()
    freq, inten = load_data(input_args)
