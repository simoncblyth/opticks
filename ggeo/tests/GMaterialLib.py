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

import os
import numpy as np

path_ = lambda _:os.path.expandvars("$IDPATH/%s" % _)
load_ = lambda _:np.load(path_(_))


def test_buffers():
    path = path_("GMaterialLib/GMaterialLib.npy")
    assert os.path.exists(path)
    os.system("ls -l %s " % path)
    buf = np.load(path)
    print "%100s %s " % (path, repr(buf.shape))
    print buf 
    return buf


if __name__ == '__main__':
    b = test_buffers()


