#!/bin/bash -l 

fold=$(dirname $0)
name=$(basename $0)
stem=${name/.sh}
script=$fold/${stem}.py

cmd="${IPYTHON:-ipython} -i -- $script $* "

echo $cmd
eval $cmd
rc=$?

echo rc $rc

