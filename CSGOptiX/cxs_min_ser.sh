#!/bin/bash
usage(){ cat << EOU
cxs_min_ser.sh
=================

Started from cxs_min_scan.sh - but multi invokation
scanning introduced reporting complications from overwrites
of SProf.txt. So instead change SER config between events in
a single real run.

EOU
}

TEST=ser_medium_scan cxs_min.sh $*

exit 0

