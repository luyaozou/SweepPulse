#!/bin/bash

echo "LINUX/MAC validation script"
# generate single sweep validation outputs
python3 ../SweepPulse.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 1 -o val_single_bg5_mode1.csv
python3 ../SweepPulse.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 1 -nobase -o val_single_bg5_mode1_nobase.csv
python3 ../SweepPulse.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 1 -spline -o val_single_bg5_mode1_spline.csv
python3 ../SweepPulse.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 2 -o val_single_bg5_mode2.csv
# generate fullband sweep validation outputs
python3 ../SweepPulse.py sample_input_fullband_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -cf sample_input_fullband_freq.dat -bg 5 -mode 1 -o val_fb_bg5_mode1.csv
python3 ../SweepPulse.py sample_input_fullband_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -cf sample_input_fullband_freq.dat -bg 5 -mode 1 -nobase -o val_fb_bg5_mode1_nobase.csv
python3 ../SweepPulse.py sample_input_fullband_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -cf sample_input_fullband_freq.dat -bg 5 -mode 1 -spline -o val_fb_bg5_mode1_spline.csv
# run python validation script
python3 validation.py
