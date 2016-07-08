@echo off
echo "Windows validation script"
rem generate single sweep validation outputs
python ..\sweep.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 1 -o val_single_bg5_mode1.csv
python ..\sweep.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 1 -nobase -o val_single_bg5_mode1_nobase.csv
python ..\sweep.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 1 -spline -o val_single_bg5_mode1_spline.csv
python ..\sweep.py sample_input_single_int.dat -win 18 -bg 5 -lo sample_input_LO.dat -mode 2 -o val_single_bg5_mode2.csv
rem generate fullband sweep validation outputs
python ..\sweep.py sample_input_fullband_int.dat -bg 5 -cf sample_input_fullband_freq.dat -lo sample_input_LO.dat -bg 5 -mode 1 -o val_fb_bg5_mode1.csv
python ..\sweep.py sample_input_fullband_int.dat -bg 5 -cf sample_input_fullband_freq.dat -lo sample_input_LO.dat -bg 5 -mode 1 -nobase -o val_fb_bg5_mode1_nobase.csv
python ..\sweep.py sample_input_fullband_int.dat -bg 5 -cf sample_input_fullband_freq.dat -lo sample_input_LO.dat -bg 5 -mode 1 -spline -o val_fb_bg5_mode1_spline.csv
rem run python validation script
python validation.py
pause
