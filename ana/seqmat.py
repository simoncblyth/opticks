#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""
seqmat.py 
=============================================

Debugging seqmat mismatch, zeros.

::

    tconcentric.py --cmx 5

    [2016-11-06 17:30:15,759] p43702 {/Users/blyth/opticks/ana/seq.py:404} INFO - compare dbgseq 0 dbgmsk 0 
    .                seqmat_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000      2325.00/233 =  9.98 
      12              4443231          3040         3272             8.53        0.929 +- 0.017        1.076 +- 0.019  [7 ] Gd Ac LS Ac MO MO MO
      40     3443231323443231           194          483           123.37        0.402 +- 0.029        2.490 +- 0.113  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO Ac
      50     4443231323443231           299           57           164.51        5.246 +- 0.303        0.191 +- 0.025  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO MO
      62     3323111323443231           181            1           178.02      181.000 +- 13.454        0.006 +- 0.006  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Gd Ac LS Ac Ac
      68     4323111323443231             0          147           147.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Gd Ac LS Ac MO
      70         344323132231           147          111             5.02        1.324 +- 0.109        0.755 +- 0.072  [12] Gd Ac LS LS Ac Gd Ac LS Ac MO MO Ac
      76     4323132344323111             0          132           132.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO
      79     3323132344323111           126            1           123.03      126.000 +- 11.225        0.008 +- 0.008  [16] Gd Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac Ac
      84     3323113234432311           118            0           118.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Ac LS Ac Ac
      86     1132231323443231           114           32            46.05        3.562 +- 0.334        0.281 +- 0.050  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS LS Ac Gd Gd
      91     1132344323443231           108           16            68.26        6.750 +- 0.650        0.148 +- 0.037  [16] Gd Ac LS Ac MO MO Ac LS Ac MO MO Ac LS Ac Gd Gd
      93     4323113234432311             0          107           107.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Ac LS Ac MO
     106     1132344323132231            84           23            34.78        3.652 +- 0.398        0.274 +- 0.057  [16] Gd Ac LS LS Ac Gd Ac LS Ac MO MO Ac LS Ac Gd Gd
     107     3132344323443231             0           83            83.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS Ac MO MO Ac LS Ac MO MO Ac LS Ac Gd Ac
     110              2223111            79           52             5.56        1.519 +- 0.171        0.658 +- 0.091  [7 ] Gd Gd Gd Ac LS LS LS
     111     3132231323443231             0           79            79.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS LS Ac Gd Ac
     125     2332332332332231             0           64            64.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS LS Ac Ac LS Ac Ac LS Ac Ac LS Ac Ac LS
     127     3322311323443231            60            0            60.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Ac LS LS Ac Ac
     129     3332332332332231            56            4            45.07       14.000 +- 1.871        0.071 +- 0.036  [16] Gd Ac LS LS Ac Ac LS Ac Ac LS Ac Ac LS Ac Ac Ac
     135     2231111323443231            51            6            35.53        8.500 +- 1.190        0.118 +- 0.048  [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Gd Gd Ac LS LS
    .                               1000000      1000000      2325.00/233 =  9.98 


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt
from opticks.ana.nbase import count_unique_sorted


if __name__ == '__main__':
    ok = opticks_main(det="concentric",src="torch",tag="1")


    #seq = "Gd Ac LS Ac MO MO MO"
    #seq = "TO BT BT BT BT DR AB"
    #seq = "TO BT BT BT BT SC AB"
    #seq = "Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO MO"
    seq = "Gd Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac Ac"

    a = Evt(tag="%s"%ok.utag, src=ok.src, det=ok.det, args=ok, seqs=[seq])
    b = Evt(tag="-%s"%ok.utag, src=ok.src, det=ok.det, args=ok, seqs=[seq])

    a.history_table(slice(0,20))
    b.history_table(slice(0,20))

    acu = count_unique_sorted(a.seqhis[a.psel])
    bcu = count_unique_sorted(b.seqhis[b.psel])




