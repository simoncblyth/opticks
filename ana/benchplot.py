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
benchplot.py
============

::

    ip benchplot.py --name geocache-bench360 --include xanalytic --include 10240,5760,1


"""

import os, logging, sys
from collections import OrderedDict as odict
import numpy as np
log = logging.getLogger(__name__)

from opticks.ana.bench import Bench
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import matplotlib.pyplot as plt

from opticks.ana.plot import init_rcParams
init_rcParams(plt)


def barplot(labels, values):
    """
    """
    iso = np.argsort( values )

    ivalues = (values*100).astype(np.int)


    cmap = plt.get_cmap('RdYlGn')( np.linspace(0.15, 0.85, 100))
    color = cmap[ivalues]

    

    widths = values
    starts = 0
    ax.barh(labels[iso], widths[iso], left=starts, height=0.5, color=color[iso])
    xcenters = starts + widths - 0.1


    fmt_ = lambda _:"%10.3f" % _  

    for y, (x, c) in enumerate(zip(xcenters, widths)):

        r, g, b, _ = color[y]
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        text_color = 'black'

        ax.text(x, y, fmt_(c), ha='center', va='center', color=text_color)
    pass
    return fig, ax


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ratios = odict()
    ratios["R0/1_TITAN_V"] = "R0_TITAN_V R1_TITAN_V".split()
    ratios["R0/1_TITAN_RTX"] = "R0_TITAN_RTX R1_TITAN_RTX".split()
    ratios["R1/0_TITAN_V"] = "R1_TITAN_V R0_TITAN_V".split()
    ratios["R1/0_TITAN_RTX"] = "R1_TITAN_RTX R0_TITAN_RTX".split()

    args = Bench.Args()
    args.ratios = ratios

    b = Bench(args)
    print(b)

    
    titles = odict()
    titles["20190526_143808"] = "JUNO360 raytrace with 1/2/4/8 NVIDIA Tesla GV100 GPUs "
    titles["20190526_202537"] = "JUNO360 raytrace with NVIDIA TITAN V and TITAN RTX GPUs"

    df = titles.keys()[1]

    rg = b.find(df)
    title = titles.get(df, "benchplot")
    xlabel = "RO:RTX OFF, R1:RTX ON    Time(s) to raytrace 10240 x 5760 (59M) pixels  "


    labels = rg.a.label
    values = rg.a.metric

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.invert_yaxis()

    plt.title(title)

    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, values.max()*1.1 )

    ax.set_xlabel( xlabel ) 


    fig, ax = barplot(labels, values)
    make_axes_area_auto_adjustable(ax)

    plt.ion()
    plt.show()

    print("savefig %s " % rg.path)
    plt.savefig(rg.path)






