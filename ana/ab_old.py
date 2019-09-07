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


class MXD(object):
    def __init__(self, ab, key, cut, erc, shortname): 
        """
        :param ab:
        :param key: property name which returns a dict with numerical values  
        :param cut: warn/error/fatal maximum permissable deviations, exceeding error level yields non-zero RC
        :param erc: integer return code if any of the values exceeds the cut 

        RC passed from python to C++ via system calls 
        are truncated beyond 0xff see: SSysTest
        """ 
        self.ab = ab 
        self.key = key
        self.cut = cut
        self.erc = erc
        self.shortname = shortname

    mxd = property(lambda self:getattr(self.ab, self.key))

    def _get_mx(self):
        mxd = self.mxd
        return max(mxd.values()) if len(mxd) > 0 else 999.
    mx = property(_get_mx)

    def _get_rc(self):
        return self.erc if self.mx > self.cut[1] else 0  
    rc = property(_get_rc)

    def __repr__(self):
        mxd = self.mxd
        pres_ = lambda d:" ".join(map(lambda kv:"%10s : %8.3g " % (kv[0], kv[1]),d.items()))  
        return "\n".join(["%s  .rc %d  .mx %7.3f .cut %7.3f/%7.3f/%7.3f   %s  " % ( self.shortname, self.rc,  self.mx, self.cut[0], self.cut[1], self.cut[2], pres_(mxd) )]) 
                       

class RC(object):
    def __init__(self, ab ):
        self.ab = ab 
        self.c2p = MXD(ab, "c2p",  ab.ok.c2max,  77, "ab.rc.c2p") 
        self.rdv = MXD(ab, "rmxs", ab.ok.rdvmax, 88, "ab.rc.rdv") 
        self.pdv = MXD(ab, "pmxs", ab.ok.pdvmax, 99, "ab.rc.pdv") 

    def _get_rcs(self):
        return map(lambda _:_.rc, [self.c2p, self.rdv, self.pdv])
    rcs = property(_get_rcs) 
        
    def _get_rc(self):
        return max(self.rcs+[0])
    rc = property(_get_rc) 

    def __repr__(self):
        return "\n".join([
                "ab.rc     .rc %3d      %r " % (self.rc, self.rcs) , 
                 repr(self.c2p),
                 repr(self.rdv),
                 repr(self.pdv),
                 "."
                  ])




