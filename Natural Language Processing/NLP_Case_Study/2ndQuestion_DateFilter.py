# -*- coding: utf-8 -*-
"""
Created on Mon May  4 03:41:15 2020

@author: rajhi
"""
import re
f = open("dataDoc.txt", "r")
read = f.read()
# Regex Pattern for identifying dates
pattern = "\d{1,2}[/-:]\d{1,2}[/-:]\d{2,4}"
print(re.findall(pattern, read))