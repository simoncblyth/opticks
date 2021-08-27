#!/bin/bash -l 

export OKG4Mgr=INFO
export CManager=INFO
export CG4=INFO
export CEventAction=INFO
export CCtx=INFO
export CWriter=INFO


which tboolean.sh
cmd="LV=box CMaterialBridge=INFO tboolean.sh --generateoverride 10000 -D"

echo $cmd
eval $cmd
rc=$?

echo rc $rc
