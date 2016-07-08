#!/bin/bash

echo "LINUX/MAC validation script"
# check python version
ret="python -c 'import sys; print("%i" % (sys.hexversion<0x03000000))'"
if [ $ret -eq 0 ]; then
    echo "python3 detected"
    # generate single sweep validation outputs
    python3 ../SweepPulse.py sample_input_single_int.dat -bg 5 -mode 1 -o val_single_bg5_mode1.csv
    python3 ../SweepPulse.py sample_input_single_int.dat -bg 5 -mode 1 -nobase -o val_single_bg5_mode1_nobase.csv
    python3 ../SweepPulse.py sample_input_single_int.dat -bg 5 -mode 1 -spline -o val_single_bg5_mode1_spline.csv
    python3 ../SweepPulse.py sample_input_single_int.dat -bg 5 -mode 2 -o val_single_bg5_mode2.csv
    # generate fullband sweep validation outputs
    python3 ../SweepPulse.py sample_input_fullband_int.dat -bg 5 -cf sample_input_fullband_freq.dat -bg 5 -mode 1 -o val_fb_bg5_mode1.csv
    python3 ../SweepPulse.py sample_input_fullband_int.dat -bg 5 -cf sample_input_fullband_freq.dat -bg 5 -mode 1 -nobase -o val_fb_bg5_mode1_nobase.csv
    python3 ../SweepPulse.py sample_input_fullband_int.dat -bg 5 -cf sample_input_fullband_freq.dat -bg 5 -mode 1 -spline -o val_fb_bg5_mode1_spline.csv
    # run python validation script
    python3 validation.py
else
    echo "python3 is required"
fi
