#!/bin/bash -l 
usage(){ cat << EOU
SPMT_scan.sh
==============

HUH getting nan again with N_MCT=900 N_SPOL=2 scan::

    In [37]: f.ARTE.squeeze()[115:120]
    Out[37]: 
    array([[[0.412, 0.317, 0.683, 0.507],
            [0.838, 0.612, 0.388, 0.507]],

           [[0.412, 0.377, 0.623, 0.507],
            [0.848, 0.667, 0.333, 0.507]],

           [[0.423, 0.499, 0.501, 0.507],
            [0.863, 0.758, 0.242, 0.507]],

           [[  nan,   nan,   nan, 0.507],
            [  nan,   nan,   nan, 0.507]],

           [[  nan,   nan,   nan, 0.507],
            [  nan,   nan,   nan, 0.507]]], dtype=float32)




EOU
}

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd )

N_MCT=900 N_SPOL=1  $REALDIR/SPMT_test.sh


