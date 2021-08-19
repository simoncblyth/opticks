#!/bin/bash -l

emms=$(seq 0 9)
for emm in $emms ; do 
    EMM=$emm, ./cxr_view.sh
done

