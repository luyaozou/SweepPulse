#!/bin/bash

echo "LINUX/MAC validation script"
## remove previous validation outputs
rm val*.csv

## generate single sweep validation outputs
# fg and bg options
python3 ../sweep.py sample_input_single_inten.dat -bdwth 18 -bg 5 -lo sample_input_fb_couple-of-lines_lo.dat -box 5 -o val_single_fg1-bg5_box5.csv
python3 ../sweep.py sample_input_single_inten.dat -bdwth 18 -fg 2 -bg 5 -lo sample_input_fb_couple-of-lines_lo.dat -o val_single_fg2-bg5.csv
python3 ../sweep.py sample_input_single_inten.dat -bdwth 18 -fg 5 -bg 1 -lo sample_input_fb_couple-of-lines_lo.dat -box 5 -o val_single_fg5-bg1_box5.csv
python3 ../sweep.py sample_input_single_inten.dat -bdwth 18 -lo sample_input_fb_couple-of-lines_lo.dat -box 5 -o val_single_avg.csv
# spline & nobase options
python3 ../sweep.py sample_input_single_inten.dat -bdwth 18 -bg 5 -lo sample_input_fb_couple-of-lines_lo.dat -nobase -o val_single_fg1-bg5_nobase.csv
python3 ../sweep.py sample_input_single_inten.dat -bdwth 18 -bg 5 -lo sample_input_fb_couple-of-lines_lo.dat -spline -o val_single_fg1-bg5_spline.csv

# generate fullband sweep validation outputs
for f in couple-of-lines numerous-lines wavybase-noline
do
  python3 ../sweep.py "sample_input_fb_$f""_inten."* -cf "sample_input_fb_$f""_freq."* -bg 5 -lo "sample_input_fb_$f""_lo."* -box 5 -o "val_fb_$f""_box5.csv"
  python3 ../sweep.py "sample_input_fb_$f""_inten."* -cf "sample_input_fb_$f""_freq."* -bg 5 -lo "sample_input_fb_$f""_lo."* -nobase -o "val_fb_$f""_nobase.csv"
  python3 ../sweep.py "sample_input_fb_$f""_inten."* -cf "sample_input_fb_$f""_freq."* -bg 5 -lo "sample_input_fb_$f""_lo."* -spline -o "val_fb_$f""_spline.csv"
done

# run python validation script
python3 validation.py
