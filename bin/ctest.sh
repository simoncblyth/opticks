#!/bin/bash

log=ctest.log 

date | tee $log
ctest $* --interactive-debug-mode 0 --output-on-failure 2>&1 | tee -a $log
date | tee -a $log



