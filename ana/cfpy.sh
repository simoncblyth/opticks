#!/bin/bash -l 

py2=/tmp/ana_py2.log
py3=/tmp/ana_py3.log

python2.7  $(which tboolean.py) --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show  2> $py2
python3    $(which tboolean.py) --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show  2> $py3

echo diff $py2 $py3
diff $py2 $py3






