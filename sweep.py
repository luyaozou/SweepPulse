# encoding = utf8
''' This script processes the linear-sweep data for pulsed experiment.
It is REQUIRED that the data takes full cycles of sweeps. '''

import argparse
import numpy as np
import re
from scipy import interpolate
from scipy.optimize import leastsq

# ------------------------------------------
# ------ MESSAGE CONSTANT DECLARATION ------
# ------------------------------------------
FILE_ERR_MSG = {0: '',                              # Silent
                1: '{:s} does not exist',           # FileNotFoundError
                2: '{:s} format is not supported',  # Format Issue
                }

# -------------------------------------------------------
# ------ Function Declaration (Alphabetical Order) ------
# -------------------------------------------------------

def analyze_fmt(file_name):
    ''' Analyze the data text format: delimiter and header

    Arguments:
    file_name -- data file name, str

    Returns:
    delm -- delimiter, str
    hd   -- number of header rows, int
    eof  -- end of file, boolean
    '''

    hd = 0
    delm = None
    a_file = open(file_name, 'r')
    # match two numbers and a delimiter
    pattern = re.compile('(-?\d+\.?\d*(e|E.?\d+)?)( |\t|,)+(-?\d+\.?\d*(e|E.?\d+)?)')

    for a_line in a_file:
        if re.match('-?\d+\.?\d*(e|E)?.?\d+ *$', a_line):
            # if the first line is a pure number
            delm = ','
            break
        else:
            try:
                delm = pattern.match(a_line).group(3)
                break
            except AttributeError:
                hd += 1

    # check if end of the file is reached
    eof = (a_file.read() == '')

    a_file.close()

    return delm, hd, eof


def avg_inten(inten, pts, sweep_num):
    ''' Average all odd intensity sweeps togethere.

    Arguments:
    inten -- intensity waveform, 1D/2D np.array
    pts   -- number of data points in a single sweep, int
    sweep_num -- number of sweeps, int

    Returns:
    inten_avg -- averaged intenisity array, 1D/2D np.array
    '''


    # Separate odd and even numbers of cycles and sum up
    if len(inten.shape)==1:
        inten_odd = np.zeros(pts)
        for i in range(sweep_num//2):
            inten_odd += inten[i*2*pts:(i*2+1)*pts]
    else:
        inten_odd = np.zeros((pts, inten.shape[1]))
        for i in range(sweep_num//2):
            inten_odd += inten[i*2*pts:(i*2+1)*pts, :]

    return inten_odd / (sweep_num//2)


def box_car(x, win):
    ''' Boxcar smooth.

    Arguments:
    x   -- x frequency array, np.array
    win -- box-car window, integer

    Returns:
    x_new -- new x array, np.array
    '''

    if win == 1:
        return x
    else:
        x_new = np.convolve(x, np.ones(win), 'valid')/win
        return x_new


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


def check_type(var):
    ''' Check data type of variables. If var is None, exit the program '''

    if isinstance(var, type(None)):
        exit()
    else:
        pass


def delay_inten(inten, delay):
    ''' Delay intensity array.

    Arguments:
    inten -- intensity array, nd.array
    delay -- delay number of points.

    Returns:
    inten_new -- delayed intensity array
    '''

    dim = inten.shape[0]
    # check the dimension of intensity array
    if len(inten.shape)==1:
        inten_new = np.roll(inten, dim - delay)
    else:
        inten_new = np.roll(inten, dim - delay, axis=0)

    return inten_new


def db_poly(y, deg=1):
    ''' Polynomial baseline clean.

    Arguments:
    y -- input array, 1D np.array

    Returns:
    y_db -- debaselined array, 1D np.array
    '''

    x = np.arange(len(y))
    popt = np.polyfit(x, y, deg)

    return y - np.polyval(popt, x)


def db_spline(y):
    ''' Background subtraction by fitting spline to the baseline.

    Arguments:
    y -- input array, 1D np.array

    Returns:
    y_db -- debaselined array, 1D np.array
    '''

    # Because of the discharge disturbance, the background does not exactly
    # match the signal. This creates a curved baseline after background
    # subtraction. Try to fit a B-spline to the baseline.
    x = np.arange(len(y))
    # Construct weight
    weight = np.exp(-np.power(y - y.max(), 2))
    # Interpolate spline
    spline = interpolate.UnivariateSpline(x, y, w=weight, k=5)

    return y - spline(x)


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


def flat_wave(freq, inten, nobase=False):
    ''' Flatten frequency and intensity arrays.

    Arguments:
    freq  -- frequency array, 1D/2D np.array
    inten -- intensity array, 1D/2D np.array
    nobase -- input argument option (do not perform intensity stitch)

    Returns:
    freq_flat  -- flattened frequency array, 1D np.array
    inten_flat -- flattened intensity array, 1D np.array
    '''

    if len(inten.shape)==1:
        return freq, inten
    else:
        # Flatten frequency and intensity matrices into vector waveforms
        # and sort frequency
        freq_flat_index = np.argsort(freq.flatten('F'))
        freq_flat = freq.flatten('F')[freq_flat_index]
        inten_flat = inten.flatten('F')[freq_flat_index]

        if not nobase:
            # reconstruct intensity matrix
            inten_recon = inten_flat.reshape(inten.shape, order='F')
            # stitch intensity
            inten = glue_sweep(inten_recon)
            # flatten intensity again
            inten_flat = inten.flatten('F')
        else:
            pass

        return freq_flat, inten_flat


def glue_sweep(y):
    ''' shift each sweep intensity to make the end of the previous sweep is
    equal to the start of the next sweep, so that the spectrum is
    continuous.

    Arguments:
    y -- intensity array, 2D np.array

    Returns:
    y_stitched -- stitched intensity array, 2D np.array
    '''

    # Get the difference of the end of col and the start of col+1
    col_shift = np.roll(y[-1, :], 1) - y[0, :]
    col_shift[0] = 0
    col_shift_accum = np.cumsum(col_shift)

    # Apply shift correction to all columns
    y_stitched = y + np.tile(col_shift_accum, (y.shape[0], 1))

    return y_stitched


def load_data(args):
    ''' Load all data files specified from input arguments. Perform background
    subtraction and data truncation according to input specifications.

    Arguments:
    args -- input argument, argparse Object

    Returen:
    freq -- frequency waveform, np.array 1D
    inten -- intensity waveform, np.array 1D/2D
    '''

    # load intensity file
    inten = load_single_file(args.inten[0])
    # exit if intensity file is not loaded correctly
    check_type(inten)

    # load lo file if available
    if args.lo:
        lo = load_single_file(args.lo[0])
        check_type(lo)
        sweep_num = np.count_nonzero(np.delete(lo*np.roll(lo, 1), 0) < 0)
        sweep_up = lo[0] < 0
    else:
        # no lo file, invoke interactive interface
        sweep_num, sweep_up = interactive()

    # number of points in a single sweep
    pts = inten.shape[0] // sweep_num

    # set bandwidth
    if args.bdwth:
        bdwth = args.bdwth[0]
    else:
        bdwth = 1.

    # reconstruct frequency array
    if args.cf:
        try:
            center_freq = float(args.cf[0])
        except ValueError:
            center_freq = load_single_file(args.cf[0])
            bdwth = center_freq[1] - center_freq[0]
    else:
        center_freq = 0

    freq = reconstr_freq(center_freq, pts, sweep_up=sweep_up, bdwth=bdwth)

    # roll the intensity array if detector delay is specified
    if args.delay:
        inten = delay_inten(inten, args.delay[0])
    else:
        pass

    if args.bg: # background subtraction if db option specified
        inten = sub_bg(inten, args.bg[0], pts)
    else:       # average intensity if db option not specified
        inten = avg_inten(inten, pts, sweep_num)

    # truncate freq & inten array if delay is specified
    if args.delay:
        freq, inten = trunc(freq, inten, args.delay[0])

    return freq, inten


def load_single_file(file_name):
    ''' Load single data file & raise exceptions.

    Arguments:
    file_name -- input file name, str

    Returns:
    data -- np.array
    '''

    try:
        delm, hd, eof = analyze_fmt(file_name)
        if eof or isinstance(delm, type(None)):
            print(err_msg_str(file_name, 2))
        else:
            data = np.loadtxt(file_name, delimiter=delm, skiprows=hd)
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
            break
        except ValueError:
            typed = input('''Must be an integer! Retype: ''').split()

    # Ask if the first sweep goes up
    typed = input('Does the first sweep go up? Y|n ')
    sweep_up = typed in ('y', 'Y', 'yes', 'Yes', 'YES')

    return sweep_num, sweep_up


def proc_nb(freq, inten, args):
    ''' Process narrow band (single sweep) according to input specifications.
        Inclues: box-car smooth in each sweep;
                 linear correction of baseline in each sweep;

    Arguments:
    freq  -- freuency array, 1D/2D np.array
    inten -- intensity array, 1D/2D np.array
    args  -- input arguments, argparse Object

    Returns:
    freq_b  -- processed frequency array, 1D/2D np.array
    inten_p/b -- processed intensity array, 1D/2D np.array
    '''

    if args.box:    # do box-car smooth
        box_win = (args.box[0])
        if len(inten.shape)==1:
            freq_b = box_car(freq, box_win)
            inten_b = box_car(inten, box_win)
        else:
            freq_b = np.apply_along_axis(box_car, 0, freq, box_win)
            inten_b = np.apply_along_axis(box_car, 0, inten, box_win)
    else:
        freq_b = freq
        inten_b = inten

    if args.nobase:     # if no baseline removal is specified
        return freq_b, inten_b
    else:
        # Apply linear correction on each sweep
        inten_p = np.apply_along_axis(db_poly, 0, inten_b, 1)
        if args.spline:
            inten_p = np.apply_along_axis(db_spline, 0, inten_b)
        return freq_b, inten_p


def proc_wb(x, y, args):
    ''' Process wide band (full stiched spectrum).

    Arguments:
    x    -- x frequency data array, 1D np.array
    y    -- y intensity data array, 1D np.array
    args -- input arguments, argparse Object

    Returns:
    y_db -- debaselined, 1D np.array
    '''

    if args.nobase:
        y_db = y
    else:
        y_db = db_poly(y, deg=1)
        if args.spline:
            y_db = db_spline(y_db)
        else:
            pass

    return y_db


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


def save_output(data, args):
    ''' Output data in csv format.

    Arguments:
    data -- output xy data, 2D np.array
    args -- input arguments, argparse Object

    Returns: None
    '''

    if args.o:
        out_name = args.o[0]
    else:
        out_name = 'SPlot_' + args.inten[0]

    np.savetxt(out_name, data, header='freq,inten', delimiter=',',
               newline='\n', fmt='%.10g', comments='')
    print('-- {:s} Saved --'.format(out_name))

    return None


def sub_bg(inten, bg, pts): # Have problems in input variables here
    ''' Background subtraction routine.

    Arguments:
    inten -- intensity waveform, 1D/2D np.array
    bg    -- ordinal number of the background sweep, int
    pts   -- number of data points in a single sweep, int

    Returns:
    inten_db -- background subtracted array, 1D/2D np.array
    '''

    if len(inten.shape)==1:     # if intensity is 1D array
        inten_sig = inten[0:pts]
        inten_bg = inten[(bg-1)*pts:bg*pts]
    else:                       # if intensity is 2D array
        inten_sig = inten[0:pts, :]
        inten_bg = inten[(bg-1)*pts:bg*pts, :]

    # Check if background is at an odd-number sweep or an even-number sweep.
    # If even, the sequency of the intensity needs to be flipped to match
    # the 1st sweep. Though it is not recommended because the waveforms are
    # not exactly the same based on experimental results.

    if not (bg % 2):  # even-number background
        inten_bg = np.flipud(inten_bg)
    else:
        pass

    return inten_sig - inten_bg


def trunc(freq, inten, delay):
    ''' Truncate frequency and intensity array if delay is specified.

    Arguments:
    freq  -- frequency array, 1D/2D np.array
    inten -- intensity array, 1D/2D np.array
    delay -- number of data points delayed in inten, inten

    Returns:
    freq_tr  -- truncated frequency, 1D/2D np.array
    inten_tr -- truncated intensity, 1D/2D np.array
    '''

    if len(inten.shape)==1:
        freq_tr = freq[:-delay]
        inten_tr = inten[delay:]
    else:
        freq_tr = freq[:-delay, :]
        inten_tr = inten[delay:, :]

    return freq_tr, inten_tr


# ---------------- Input Argument Parser ----------------
def arg():
    ''' Input arguments parser. Returns: argparse Object.'''

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

    return parser.parse_args()


# ---------------- main routine ----------------
if __name__ == '__main__':

    input_args = arg()
    freq, inten = load_data(input_args)
    freq, inten = proc_nb(freq, inten, input_args)
    freq_flat, inten_flat = flat_wave(freq, inten, input_args.nobase)
    inten_flat = proc_wb(freq_flat, inten_flat, input_args)
    save_output(np.column_stack((freq_flat, inten_flat)), input_args)
