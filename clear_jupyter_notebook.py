#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

current_dir = os.getcwd()

list_of_files = []
for (dirpath, dirnames, filenames) in os.walk(current_dir):
    list_of_files += [os.path.join(dirpath, file) for file in filenames]

list_of_ipynb_files = [file for file in list_of_files if file.endswith(".ipynb")]

for file in list_of_ipynb_files:
    os.system(f"jupyter nbconvert --clear-output --inplace {file}")

print("All notebooks have been cleared of their outputs.")

