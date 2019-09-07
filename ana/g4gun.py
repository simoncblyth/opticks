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
g4gun.py: loads G4Gun event
===============================

To create the event use::

   ggv-
   ggv-g4gun


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)


from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt

import matplotlib.pyplot as plt


mnavmx_ = lambda _:(np.min(_),np.average(_),np.max(_))


def red_gensteps_everywhere_issue(gs):
    """
    """
    post = gs[:,1] 
    print post.shape
    print post

    npho = gs[:,0,3].view(np.int32)  ## approx half the gs have zero photons ??

    npho_0 = np.count_nonzero(npho == 0)
    npho_1 = np.count_nonzero(npho == 1)
    npho_2 = np.count_nonzero(npho > 1)


    cx = np.array(map(mnavmx_, map(np.array, post.T )))
    ce = cx[:3,1]
    print cx 

    x = post[:,0] - ce[0]
    y = post[:,1] - ce[1]
    z = post[:,2] - ce[2]
    t = post[:,3]

    #s = t < 5
    s = npho > 0

    fig = plt.figure()

    ax = fig.add_subplot(221)
    ax.hist(x[s], bins=100)
    ax.set_yscale('log')

    ax = fig.add_subplot(222)
    ax.hist(y[s], bins=100)
    ax.set_yscale('log')

    ax = fig.add_subplot(223)
    ax.hist(z[s], bins=100)
    ax.set_yscale('log')

    ax = fig.add_subplot(224)
    ax.hist(t[s], bins=100)
    ax.set_yscale('log')
    fig.show()



if __name__ == '__main__':

    ok = opticks_main(src="G4Gun", det="G4Gun", tag="-1")

    try:
        evt = Evt(tag=ok.tag, src=ok.src, det=ok.det, args=ok)
    except IOError as err:
        log.fatal(err)
        sys.exit(ok.mrc) 
    pass

    print evt

    red_gensteps_everywhere_issue(evt.gs)




   

