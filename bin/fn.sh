#!/usr/bin/env bash -l

fn-

cmd="fn-lv $*"

echo $cmd
eval $cmd
rc=$?

echo $0 rc $rc
exit $rc

