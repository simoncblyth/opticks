#!/bin/bash -l

./check.sh 
[ $? -ne 0 ] && echo check failed && exit 1

which CerenkovMinimal
CerenkovMinimal 

