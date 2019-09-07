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
Lines.py
=========

Tool for operating on complex lines of output, such as those from a verbose CMake build.

::

    cat /tmp/thrap-expandTest-i | Lines.py --starting -    

         ## gives overview of CMake actions 

    cat /tmp/thrap-expandTest-i | Lines.py --after "-- Generating /" --splitlong     

         ## focus on area of interest

    cat /tmp/thrap-expandTest-i | Lines.py --after "-- Generating /" --splitlong --slice 6:7  

         ## dump one source line only split across many lines for legibility   
     
    cd /usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/tests    
    cat /tmp/thrap-expandTest-i | Lines.py --after "-- Generating /" --splitlong --slice 6:7 | sh    

         ## rerun the command 
         ## commands using relative paths will need a cd to the appropriate directory first 


"""
import sys, re, os, logging, argparse
log = logging.getLogger(__name__)

class Line(object):
    def __init__(self, idx, txt, splitlong=False, longcut=150):
        self.idx = idx
        self.txt = txt
        self.is_split = splitlong and len(txt) > longcut 
        self.ltxt = txt.replace(" "," \\\n")

    hdr = property(lambda self:"[%3d;%3d]" % ( self.idx, len(self.txt) ))

    def __str__(self):
        return self.ltxt if self.is_split else self.txt
    def __repr__(self):
        return "%s %s " % ( self.hdr, self.txt) 


class Lines(object):
    def __init__(self, srclines, args):
        self.args = args
        self.lines = []

        for idx,txt in enumerate(srclines):
            l = Line(idx+1, txt, args.splitlong, args.longcut)
            if l.is_split:
                self.lines.append(Line(0,"##( %d \n" % idx))
            pass
            self.lines.append(l)
            if l.is_split:
                self.lines.append(Line(0,"\n##)"))
            pass
        pass
    def __str__(self):
        return "\n".join(map(str, self.lines))


if __name__ == '__main__':


    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "--after", help="Start the line index count from the line after the first line that starts with the string provided." )
    parser.add_argument(     "--slice", help="Line slice to operate upon starting from the --after start inded. Slice specified in python range string format, eg \"0:1\" or \"3:5\". Default %(default)s. " )
    parser.add_argument(     "--starting", help="Filter lines operated upon, only outputting ones starting with the string provided." )
    parser.add_argument(     "--splitlong", help="Split long lines operated upon", action="store_true" )
    parser.add_argument(     "--longcut", type=int, default=150, help="Number of characters for a long line, that may be split. Default %(default)s." )

    args = parser.parse_args()
    args.slice_ = slice(*map(int, args.slice.split(":"))) if args.slice is not None else slice(0,None,None)


    lines_input = map(str.rstrip, sys.stdin.readlines() ) 
    lines_filtered = filter( lambda _:_.startswith(args.starting), lines_input) if args.starting is not None else lines_input

    if args.after is not None:
        matchlines = filter( lambda _:_[1].startswith(args.after), enumerate(lines_filtered))
        assert len(matchlines) > 0, "--after argument %s failed to match any input lines " % ( args.after )
        start = matchlines[0][0]         
    else:
        start = 0 
    pass 

    lines_after = lines_filtered[start:]
    lines_range = lines_after[args.slice_]

    ll = Lines(lines_range, args)
    print str(ll)


 

