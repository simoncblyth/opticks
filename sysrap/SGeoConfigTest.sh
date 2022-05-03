#!/bin/bash -l 


arglist=/tmp/arglist.txt
cat << EOL > $arglist
red
green
blue
cyan
magenta
yellow
black
EOL

OPTICKS_ARGLIST_PATH=$arglist SGeoConfigTest 


