#!/bin/bash

echo "LINUX/MAC validation script"
# remove previous validation outputs
rm val*.csv
# generate single sweep validation outputs
python3 ../sweep.py sample_input_single_int.dat -bdwth 18 -bg 5 -lo sample_input_LO.dat -o val_single_bg5_mode1.csv
python3 ../sweep.py sample_input_single_int.dat -bdwth 18 -bg 5 -lo sample_input_LO.dat -nobase -o val_single_bg5_mode1_nobase.csv
python3 ../sweep.py sample_input_single_int.dat -bdwth 18 -bg 5 -lo sample_input_LO.dat -spline -o val_single_bg5_mode1_spline.csv
#python3 ../sweep.py sample_input_single_int.dat -bdwth 18 -bg 5 -lo sample_input_LO.dat -mode 2 -o val_single_bg5_mode2.csv
# generate fullband sweep validation outputs
python3 ../sweep.py sample_input_fullband_int.dat -bdwth 18 -bg 5 -lo sample_input_LO.dat -cf sample_input_fullband_freq.dat -bg 5 -o val_fb_bg5_mode1.csv
python3 ../sweep.py sample_input_fullband_int.dat -bdwth 18 -bg 5 -lo sample_input_LO.dat -cf sample_input_fullband_freq.dat -bg 5 -nobase -o val_fb_bg5_mode1_nobase.csv
python3 ../sweep.py sample_input_fullband_int.dat -bdwth 18 -bg 5 -lo sample_input_LO.dat -cf sample_input_fullband_freq.dat -bg 5 -spline -o val_fb_bg5_mode1_spline.csv
# run python validation script
python3 validation.py
