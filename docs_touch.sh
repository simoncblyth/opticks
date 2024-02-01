#!/bin/bash -l 

cd $(dirname $(realpath $BASH_SOURCE)) 
find docs -name '*.rst' -exec touch {} \;



