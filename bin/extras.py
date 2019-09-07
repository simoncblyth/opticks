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


from opticks.ana.base import opticks_main, dump_extras_meta
       
if __name__ == '__main__':
    args = opticks_main()
    base = "$OPTICKS_IDFOLD/extras"
    dump_extras_meta(base, name="meta.json", fmt=" %(idx)5s : %(height)6s : %(lvname)-40s : %(soname)-40s : %(err)s ")

   
    

