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


import os, sys, re, argparse, logging
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

class STrace(object):
    ptn = re.compile("^open\((?P<args>.*)\)\s*=\s*(?P<rc>\d+).*$")
    """
    open("/tmp/tboolean-box/evt/tboolean-box/torch/1/Material_SequenceLocal.json", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 54
    """ 
    def __init__(self, args):
        lines = file(args.path).read().split("\n")
        opens = []  
        for i, line in enumerate(lines):
            m = self.ptn.match(line)
            if m:
                openargs = map(str.strip,m.groupdict()["args"].split(",")) 
                if len(openargs) == 3:  
                    path, flags, mode = openargs 
                elif len(openargs) == 2:
                    path, flags = openargs 
                    mode = -1  
                else:
                    assert 0 
                pass
                assert path[0] == "\"" and path[-1] == "\"" 

                opens.append(dict(path=path[1:-1], flags=flags, mode=mode))
            pass
        pass
        self.opens = opens
        self.args = args

    def __str__(self):
        opens = self.opens 
        if self.args.flagfilter is not None:
            opens = filter(lambda d:d['flags'].find(self.args.flagfilter) > -1, opens)
        pass 
        return "\n".join(map(lambda d:" %(path)-80s : %(flags)25s : %(mode)5s " % d, opens)) 

    def __repr__(self):
        return "\n".join(["strace.py %s" % self.path] + map(lambda kv:"%4d : %s " %  (kv[1],kv[0]), self.d.items()))
     

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "path", nargs='?', default="/tmp/strace.log", help="Strace logfile, eg created with \"strace -o /tmp/strace.log -e open /path/to/executable args\" " )
    parser.add_argument(     "--exclude", default=None, help="Dont collect from lines containing the string provided" )
    parser.add_argument( "-f", "--flagfilter", default="O_CREAT", help="Report only lines containing the string provided in the flags eg O_CREAT. Default %(default)s " )
    args = parser.parse_args()
    cmdline = " ".join(["strace.py"]+sys.argv[1:])
    print(cmdline)

    st = STrace(args)

    print(st)



 


