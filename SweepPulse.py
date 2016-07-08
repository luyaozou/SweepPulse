# encoding = utf8
''' This script processes the linear-sweep data for pulsed experiment.
It is REQUIRED that the data takes full cycles of sweeps. '''

import argparse
import numpy as np
from scipy import interpolate
from scipy.optimize import leastsq

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

# --------------------------
# ------ Input Parser ------
# --------------------------

# parse arguments from terminal input
parser = argparse.ArgumentParser(description=__doc__,
                                 epilog='--- Luyao Zou, April 2015 ---')
parser.add_argument('inten', nargs=1, help='Intensity data file')
parser.add_argument('-bg', nargs=1, type=int,
                    help='''The ordinal number of the full sweep to use
                          as background. If this argument is not specified,
                          assume no background subtraction is necessary
                          and all sweep cycles are averaged together. ''')
parser.add_argument('-cf', nargs=1,
                    help='''Single center frequency (MHz) or a file listing
                          several center frequencies. If neither specified,
                          assume it's a single frequency data and set at 0. ''')
parser.add_argument('-win', nargs=1, type=float,
                    help='''Full frequency sweep window (MHz). If -win is 
                            not specified while freq file is available,
                            get sweep window from the difference of the
                            first two data points in the freq file, assuming
                            frequency is evenly spaced and matches window.''')
parser.add_argument('-box', nargs=1, type=int,
                    help='''Boxcar smooth window. Must be an odd integer.
                            Optional''')
parser.add_argument('-delay', nargs=1, type=int,
                    help=''' Delay of detector response by number of
                             data points. Default is 0 ''')
parser.add_argument('-o', nargs=1, help='Output file name. Optional')
parser.add_argument('-mode', nargs=1, type=int,
                    help='''Sweep Mode: How does the script use the LO file.

                            [ 0 ]: No LO file. Interactive input questions
                               will pop up. DEFAULT w/o LO

                            [ 1 ]: Partial LO usage. Only use LO file to
                               determine the number of full sweeps, and
                               the phase of the first sweep (up or down).
                               Assumes linear sweep and synthesize frequency.
                               DEFAULT w/ LO

                            [ 2 ]: Full LO usage. Use exact LO data points to 
                               map time to frequency. Accepts any waveforms.
                               May create non-even distribution of frequency
                               sampling due to the LO voltage variation.''')
parser.add_argument('-lo', nargs=1,
                    help='LO file. Not required if sweep mode is 0')
parser.add_argument('-spline', action='store_true',
                    help='Fit spline to subtract baseline. Optional')
parser.add_argument('-nobase', action='store_true',
                    help='Disable ALL baseline removal functionality. Optional')
parser.add_argument('-diff', action='store_true',
                    help='Take derivative to subtract baseline. Optional')

args = parser.parse_args()


# ----------------------------------------------
# ------ Read Data File & Input Arguments ------
# ----------------------------------------------

try:
    inten = np.loadtxt(args.inten[0], delimiter=',')
except FileNotFoundError:
    print('{:s} Not Found'.format(args.inten[0]))
    exit()

# Get background ordinal number
if args.bg:
    bg = args.bg[0]
else:
    bg = False

# Get detector response delay
if args.delay:
    delay = args.delay[0]
else:
    delay = False

# Get center frequency. If not specified, set as 0. If is a single number,
# convert it. Otherwise it is a file name string, so import the file.
if args.cf:
    try:    # try if it is a float number
        center_freq = float(args.cf[0])
    except ValueError:      # get file name and import the frequency vector
        center_freq = np.loadtxt(args.cf[0])
else:
    center_freq = float(0)

if type(center_freq) is float:
    single_freq = True
else:
    single_freq = False

# Get sweep window
try:
    window = args.win[0]
except TypeError:
    if single_freq:
        while True:
            try:
                window = input('''Sweep Window Not Specified! 
                                  Type in the window: ''').split()
                window = abs(float(window[0]))
                break
            except ValueError:
                window = input('Must be a number! Retype: ').split()
    else:
        # if not single_freq: frequency file is available
        # freq diff of the first 2 points
        window = center_freq[1] - center_freq[0]


# Get sweep mode
if not args.mode:
    if not args.lo:
        smode = 0
    else:
        smode = 1
else:
    smode = args.mode[0]

if not smode:  # If sweep mode is 0 or not specified
    while True:     # Get number of sweeps from user input & Error handling
        try:
            sweep_input = input('Input number of full sweeps in the data: '
                                ).split()
            sweep = int(sweep_input[0])
            break
        except ValueError:
            sweep_input = input('''Must be an integer! Retype: ''').split()
    
    sweep_input = input('Does the first sweep go up? Y|n ')
    # Ask if the first sweep goes up
    sweep_up = sweep_input in ('y', 'Y', 'yes', 'Yes', 'YES')
else:               # Other than sweep mode 0, use LO file
    lo = np.loadtxt(args.lo[0])         # Load LO file
    # Get number of sweeps by looking at the number of points crossing 0
    # in the LO file. This is done by multiplying the adjacent points in
    # the LO file and count for negative points.
    sweep = np.count_nonzero(np.delete(lo*np.roll(lo, 1), 0) < 0)
    if smode == 1:      # Mode 1: Partially use LO
        sweep_up = lo[0] < 0
    else:                    # Mode 2: Fully use LO
        freq = lo/lo.ptp()*window     # Map frequency with LO points.

# number of points in a full sweep
sweep_range = inten.shape[0]//sweep

# ---------------------------------------------
# ------ Shape Freq & Intensity Matrices ------
# ---------------------------------------------

if smode < 2:      # If sweep mode < 2, construnct frequency vector
# freq(array) = 
# center_freq +/- win_freq/2 -/+ win_freq*[0,1,...,sweep_range]/sweep_range
    if sweep_up:    # tells the script its freq high-to-low from terminal input
        freq = window*np.arange(sweep_range)/(sweep_range) - window/2
    else:
        freq = -window*np.arange(sweep_range)/(sweep_range) + window/2

    if single_freq:
        freq = freq + center_freq
    else:
        freq = np.tile(freq, (len(center_freq), 1)).transpose() + \
               np.tile(center_freq, (sweep_range, 1))
else:
    freq = freq.reshape((sweep_range, sweep), order='F')
    if single_freq:
        freq = freq[:,0]
    else:
        freq = freq[:,0] + np.tile(center_freq, (sweep_range, 1))

# Check if background subtraction is needed
if bg:
    # Extract signal and background
    freq, inten_signal, inten_bg = extract_sig_bg(freq, inten, bg, sweep,
                                                  delay, single_freq)
    # Subtract background
    inten_db = inten_signal - inten_bg
else:
    # Average all sweep cycles togethere.
    # roll the last sweep back to the first place
    if delay:
        inten = np.roll(inten, sweep_range - delay, axis=0)
    else:
        inten = np.roll(inten, sweep_range, axis=0)
    # Now remember the frequency of the first sweep cycle goes backwards
    # against the 'freq' vector
    # Separate odd and even numbers of cycles and sum up 
    if single_freq:
        inten_odd = np.zeros(sweep_range)
        inten_even = np.zeros(sweep_range)
        for i in range(sweep//2):
            inten_odd += inten[i*2*sweep_range:(i*2+1)*sweep_range]
            inten_even += inten[(i*2+1)*sweep_range:(i*2+2)*sweep_range]
    else:
        inten_odd = np.zeros((sweep_range, freq.shape[1]))
        inten_even = np.zeros((sweep_range, freq.shape[1]))
        for i in range(sweep//2):
            inten_odd += inten[i*2*sweep_range:(i*2+1)*sweep_range,:]
            inten_even += inten[(i*2+1)*sweep_range:(i*2+2)*sweep_range,:]
    # flip odd number of cycles to get the frequency right
    inten_odd = np.flipud(inten_odd)
    inten_db = (inten_odd + inten_even)/sweep

# save the matrix shape of intensity for future use
inten_shape = inten_db.shape


# ---------------------------------------------------
# ------ Map Frequency and Subtract Background ------
# ---------------------------------------------------

if not args.nobase:
    # Apply linear correction on each sweep
    inten_db = np.apply_along_axis(db_poly, 0, inten_db, 1)
# !init = [a, t, phi, m, b]
#init = [np.std(inten_db), 2, 0]
#inten_db = np.apply_along_axis(db_sinusoidal, 0, inten_db, init)

# Flatten frequency and intensity matrices into vector waveforms
# and sort frequency
freq_flat_index = np.argsort(freq.flatten('F'))
freq_flat = freq.flatten('F')[freq_flat_index]

# Reconstruct intensity in matrix form
freq = freq_flat.reshape(inten_shape, order='F')
inten_flat = inten_db.flatten('F')[freq_flat_index]
inten_db = inten_flat.reshape(inten_shape, order='F')
    
if not (single_freq or args.nobase):
    # remove jumps at junctions
    inten_db = glue_sweep(inten_db)


# ------------------------------------------------------
# --- Perform User-specified Baseline Cleanning Mode ---
# ------------------------------------------------------

# Spline baseline removal in each sweep if speficied
if not args.nobase and args.spline:
    inten_db = np.apply_along_axis(db_spline, 0, inten_db)

# ----------------------------------
# ------ Apply Box-car Smooth ------
# ----------------------------------

# Apply box-car smooth on each sweep first if specified
if args.box:
    box_win = abs(args.box[0])  # In case user input negative number
    if not (box_win % 2):
        box_win += 1    # in case it's an even number, plus one
    if box_win > 1:
        if single_freq:
            freq_out = freq[box_win//2:-box_win//2+1]
            inten_out = box_smooth(inten_db, box_win)
        else:
            # Apply boxcar smooth
            freq = freq_flat.reshape(inten_shape, order='F')
            freq = freq[box_win//2:-box_win//2+1, :]
            inten_db = np.apply_along_axis(box_smooth, 0, inten_db, box_win)
    else:
        freq_out, inten_out = freq, inten_db
else:
    freq_out, inten_out = freq, inten_db

if not single_freq:
    # Flattern matricies
    if not args.nobase:
        inten_db = glue_sweep(inten_db)
    inten_out = inten_db.flatten('F')
    freq_out = freq.flatten('F')


# -------------------------------------
# ------ Remove Big Wavy Feature ------
# -------------------------------------

if not args.nobase:
    inten_out = db_poly(inten_out, deg=1)
    if args.spline:
        inten_out = db_spline(inten_out)


# -------------------------------------------
# ---- Perform User-specified Derivative ----
# -------------------------------------------

# Taking derivative if specified
if args.diff:
    freq_out, inten_out = db_diff(freq_out, inten_out)


# -----------------------------------
# ------ Processed Data Output ------
# -----------------------------------
# Output processed file with specified headers and format.
if args.o:
    out_name = args.o[0]
else:
    out_name = 'SPlot_' + args.inten[0]
spec = np.column_stack((freq_out, inten_out))
np.savetxt(out_name, spec, header='freq,inten', delimiter=',',
           newline='\n', fmt='%.10g', comments='')
print('-- {:s} Saved --'.format(out_name))
