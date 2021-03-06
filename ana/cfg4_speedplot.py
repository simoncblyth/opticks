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
cfg4_speedplot.py 
======================


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from opticks.ana.metadata import Metadata
from opticks.ana.catdir import Catdir



def speedplot(cat, tag, a, landscape=False, ylim=None, log_=False):

    if a is None:
        log.warning("no metadata skipping")
        return   

    nnp = len(np.unique(a.numPhotons))

    if nnp != 1:
        log.fatal("numPhotons not unique, cannot compare : nnp %s " % nnp)
        log.fatal("Tags and negated counterparts should always have the same photon statistics")
        log.fatal(" TO AVOID THIS PROBLEM ADOPT A NEW TAG WHEN CHANGING PHOTON STATS ")

    assert nnp == 1, "Tags and negated counterparts should always have the same photon statistics" 

    mega = float(a.numPhotons[0])/1e6
    title = "Propagate times (s) for %3.1fM Photons with %s geometry, tag %s, [max/avg/min]" % (mega, cat, tag)  

    plt.close()
    plt.ion()

    fig = plt.figure()
    fig.suptitle(title)

    compute = a.flgs & Metadata.COMPUTE != 0 
    interop = a.flgs & Metadata.INTEROP != 0 
    cfg4    = a.flgs & Metadata.CFG4 != 0 

    msks = [cfg4, interop, compute]
    ylims = [[0,60],[0,5],[0,1]]
    labels = ["CfGeant4", "Opticks Interop", "Opticks Compute"]

    n = len(msks)
    for i, msk in enumerate(msks):

        if landscape: 
            ax = fig.add_subplot(1,n,i+1)
        else:
            ax = fig.add_subplot(n,1,i+1)
        pass
        d = a[msk]

        t = d.propagate

        mn = t.min()
        mx = t.max()
        av = np.average(t)        

        label = "%s [%5.2f/%5.2f/%5.2f] " % (labels[i], mx,av,mn)
 
        loc = "lower right" if i == 0 else "upper right" 

        ax.plot( d.index, d.propagate, "o")
        ax.plot( d.index, d.propagate, drawstyle="steps", label=label)

        if log_:
            ax.set_yscale("log")

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(ylims[i])
        pass
        ax.legend(loc=loc)
    pass


    ax.set_xlabel('All times from: MacBook Pro (2013), NVIDIA GeForce GT 750M 2048 MB (384 cores)')
    ax.xaxis.set_label_coords(-0.5, -0.07 )

    plt.show()




if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main() 

    cat = Catdir(ok.catdir)
    a = cat.times(ok.tag)


if 1:
    speedplot(cat, ok.tag, a, landscape=True, ylim=[0.1, 60], log_=True)
    

