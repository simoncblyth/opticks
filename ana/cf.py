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


import os, sys, logging, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.cfh import CFH
from opticks.ana.nbase import chi2, vnorm
from opticks.ana.decompression import decompression_bins
from opticks.ana.histype import HisType
from opticks.ana.mattype import MatType
from opticks.ana.evt import Evt
log = logging.getLogger(__name__)


class CF(object):
    def __init__(self, args, seqs=[], spawn=None, top=True):
        """
        :param args:
        :param seqs: used beneath top level 
        :param spawn:  only used from top level cf
        """

        log.info("CF.__init__ START")
        self.args = args 
        self.seqs = seqs
        self.top = top

        self.af = HisType()
        self.mt = MatType()

        self.loadevt(seqs)
        self.compare()

        self.ss = []
        self.init_spawn(spawn)
        log.info("CF.__init__ DONE")


    def init_spawn(self, spawn, flv="seqhis"):
        """
        Spawn CF for each of the selections, according to 
        slices of the history sequences.
        """
        if spawn is None:
            return 

        assert self.top == True, "spawn is only allowed at top level "
        totrec = 0 
        labels = self.seqlabels(spawn, flv)
        for label in labels:
            seqs = [label]
            scf = self.spawn(seqs)
            totrec += scf.nrec() 
            self.ss.append(scf) 
        pass
        self.totrec = totrec

    def spawn(self, seqs):
        scf = CF(self.args, seqs, spawn=None, top=False)
        scf.parent = self
        scf.his = self.his
        scf.mat = self.mat
        return scf

    def __repr__(self):
        return "CF(%s,%s,%s,%s) " % (self.args.tag, self.args.src, self.args.det, repr(self.seqs))

    def dump_ranges(self, i):
        log.info("%s : dump_ranges %s " % (repr(self), i) )

        a = self.a
        b = self.b

        ap = a.rpost_(i)
        ar = vnorm(ap[:,:2])
        if len(ar)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (ar.min(),ar.max())))

        bp = b.rpost_(0)
        br = vnorm(bp[:,:2])
        if len(br)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (br.min(),br.max())))

    def dump(self):
        self.dump_ranges(0)
        self.dump_histories()


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    args = opticks_main(tag="1", src="torch", det="default")
    log.info(" args %s " % repr(args))

    seqhis_select = slice(1,2)
    #seqhis_select = slice(0,8)
    try:
        cf = CF(tag=args.tag, src=args.src, det=args.det, select=seqhis_select )
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    cf.dump()
 
