#!/bin/bash -l 

which tboolean.sh
cmd="LV=box CMaterialBridge=INFO tboolean.sh --generateoverride 10000 -D"

echo $cmd
eval $cmd
rc=$?

echo rc $rc
