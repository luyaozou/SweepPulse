@echo off
echo "Windows unittest script"

echo ""
echo "***** Test algorithm *****"
python -m unittest unit_test/algorithm_test.py

echo ""
echo "***** Test file handling *****"
python -m unittest unit_test/fun_file_test.py

pause
