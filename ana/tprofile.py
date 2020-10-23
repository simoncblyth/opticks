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

::

    ip tprofile.py 

::

    ip () 
    { 
        local py=${1:-dummy.py};
        shift;
        ipython --pdb $(which $py) -i $*
    }


"""
from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from opticks.ana.main import opticks_main
from opticks.ana.profile_ import Profile 

if __name__ == '__main__':
    ok = opticks_main(doc=__doc__)  
    log.info(ok.brief)

    op = Profile(ok) 
    op.deltaVM()

    a = op.a  
    l = op.l 

    plt.plot( op.t, op.v, 'o' )
    plt.ion()
    plt.show()

    print(op)


