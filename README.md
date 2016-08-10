# SweepPulse

---------- v2 release ----------

New version of SweepPulse is released!

Things updated:
- Improve frequency restoration.
- Improve spline baseline removal algorithm.
- Improve data loading algorithm.
  Support multiple delimited text as well as Python Numpy binary format .npy.
- Modify input options.
  Delete useless "mode" option in v1.
  Support both foreground and background sweep selection.
- Delete v1 sample files, add v2 sample files.
- Significantly improve code readability by refactoring.


---------- v1 release ----------

Data process script for the "fast sweep" technique described in J. Phys. Chem. A, 2016, 120, 657
The experimental technique is described in detail in the publication.
The script takes the experimentally measured frequency and intensity txt files
and reconstruct the spectrum.

Usage: run sweep.py with python in local terminal.
Help message: run "python sweep.py -h"

Package Requirements:
python3
Numpy
scipy

Development:
Seeking better baseline removal algorithms.

After implementing new functionalities, run unittests and validation_test.
unit test: run shell script "run_unittest.sh"
          or manually run two python test scripts in the unit_test folder
validation test: run shell script "run_val_SHELL.sh"
                 or batch script "run_val_WIN.bat"

If key algorithms are modified or improved, the output may be different from the
samples provided.
Careful examination of new results is required before adding into the validation
sample.

Update test files and documentation after major upgrade.
