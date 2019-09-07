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


import os, numpy as np

try: 
    from hashlib import md5 
except ImportError: 
    from md5 import md5 


from opticks.ana.nbase import array_digest 



def test_hello():
    s = 'hello'
    dig = md5()
    dig.update(s)
    print s, dig.hexdigest()


def test_array_digest():
    """
    digest on the file includes the header, but array_digest 
    covers just the data
    """
    i = np.eye(4, dtype=np.float32)
    a = np.vstack([i,i,i]).reshape(-1,4,4)
    print array_digest(a)
    np.save(os.path.expandvars("$TMP/test_array_digest.npy"), a )



if __name__ == '__main__':
    test_hello()
    test_array_digest()





    
