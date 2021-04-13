#!/bin/bash -l

name=$(basename $0)
stem=${name/.sh}

usage(){ cat << EOU
Grab inputs from remote note P with::

   cd ~/j
   ./PMTEfficiencyCheck.sh grab 

EOU
}

cmd="ipython -i $stem.py"
echo $cmd
eval $cmd

