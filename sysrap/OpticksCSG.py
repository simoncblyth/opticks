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

# generated Sun Aug 13 10:06:00 2017 
# from /Users/blyth/opticks/sysrap 
# base OpticksCSG.h stem OpticksCSG 
# with command :  /Users/blyth/opticks/bin/c_enums_to_python.py OpticksCSG.h 
#0
class CSG_(object):
    ZERO = 0
    UNION = 1
    INTERSECTION = 2
    DIFFERENCE = 3
    PARTLIST = 4
    SPHERE = 5
    BOX = 6
    ZSPHERE = 7
    ZLENS = 8
    PMT = 9
    PRISM = 10
    TUBS = 11
    CYLINDER = 12
    SLAB = 13
    PLANE = 14
    CONE = 15
    MULTICONE = 16
    BOX3 = 17
    TRAPEZOID = 18
    CONVEXPOLYHEDRON = 19
    DISC = 20
    SEGMENT = 21
    ELLIPSOID = 22
    TORUS = 23
    HYPERBOLOID = 24
    CUBIC = 25
    UNDEFINED = 26
    FLAGPARTLIST = 100
    FLAGNODETREE = 101
    FLAGINVISIBLE = 102
    D2V={'pmt': 9, 'cylinder': 12, 'convexpolyhedron': 19, 'zsphere': 7, 'sphere': 5, 'zero': 0, 'disc': 20, 'cone': 15, 'hyperboloid': 24, 'slab': 13, 'flaginvisible': 102, 'intersection': 2, 'zlens': 8, 'ellipsoid': 22, 'union': 1, 'prism': 10, 'partlist': 4, 'tubs': 11, 'cubic': 25, 'plane': 14, 'multicone': 16, 'difference': 3, 'segment': 21, 'box3': 17, 'box': 6, 'undefined': 26, 'torus': 23, 'trapezoid': 18, 'flagnodetree': 101, 'flagpartlist': 100}


    @classmethod
    def raw_enum(cls):
        return list(filter(lambda kv:type(kv[1]) is int,cls.__dict__.items()))

    @classmethod
    def enum(cls):
        return cls.D2V.items() if len(cls.D2V) > 0 else cls.raw_enum()

    @classmethod
    def desc(cls, typ):
        kvs = list(filter(lambda kv:kv[1] == typ, cls.enum()))
        return kvs[0][0] if len(kvs) == 1 else "UNKNOWN"

    @classmethod
    def descmask(cls, typ):
        kvs = list(filter(lambda kv:kv[1] & typ, cls.enum())) 
        return ",".join(map(lambda kv:kv[0], kvs))

    @classmethod
    def fromdesc(cls, label):
        kvs = list(filter(lambda kv:kv[0] == label, cls.enum()))
        return kvs[0][1] if len(kvs) == 1 else -1



