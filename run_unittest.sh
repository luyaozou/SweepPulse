#!/bin/bash
echo "Unix unit tests"

echo ""
echo "***** Test algorithm *****"
python3 -m unittest unit_test/algorithm_test.py

echo ""
echo "***** Test file handling *****"
python3 -m unittest unit_test/fun_file_test.py
