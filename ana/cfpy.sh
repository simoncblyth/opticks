#!/bin/bash -l 

mode=${1:-1}

py2=/tmp/ana_py2_$mode.log
py3=/tmp/ana_py3_$mode.log

if [ "$mode" == "0" ]; then 

   python2.7  $(which tboolean.py) --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show  2> $py2
   python3    $(which tboolean.py) --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show  2> $py3

elif [ "$mode" == "1" ]; then

   LV=box python2.7 histype.py 2> $py2 1>&2
   LV=box python3 histype.py   2> $py3 1>&2

fi 

echo diff $py2 $py3
diff $py2 $py3






