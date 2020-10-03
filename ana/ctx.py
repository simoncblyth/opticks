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

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import sys, codecs
if sys.version_info.major > 2:
    u_ = lambda _:_                            # py3 strings are unicode already 
    b_ = lambda _:codecs.latin_1_encode(_)[0]  # from py3 unicode string to bytes
    d_ = lambda _:codecs.latin_1_decode(_)[0]  # from bytes to py3 unicode string
else:
    u_ = lambda _:unicode(_, "utf-8")          # py2 strings are bytes
    b_ = lambda _:_
    d_ = lambda _:_
pass
lfilter = lambda *args:list(filter(*args))
lmap = lambda *args:list(map(*args))



class Ctx(dict):
    """
    Utility providing conversions between various ways of addressing comparison histograms
    """
    DET = "concentric"
    TAG = "1"
    BASE = "$TMP/CFH"
    SEQ0 = "TO AB"
    IREC = 0

    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa) 

    QWNS = os.environ.get("OPTICKS_MAIN_QWNS", "XYZTABCR")
    qwn = property(lambda self:self.get("qwn",self.QWNS))
    det = property(lambda self:self.get("det",self.DET))
    tag = property(lambda self:self.get("tag",self.TAG))
    irec = property(lambda self:int(self.get("irec",self.IREC)))
    srec = property(lambda self:self.srec_(self.irec))
    seq0 = property(lambda self:self.get("seq0",self.SEQ0))   # seq0 is space delimited eg "TO AB"
    sqs = property(lambda self:self.seq0.split(" "))
    sseq0 = property(lambda self:"_".join(self.sqs))
    qctx = property(lambda self:os.path.join(self.sseq0,self.srec,self.qwn))
    pctx = property(lambda self:os.path.join(self.det,self.tag,self.sseq0,self.srec,self.qwn))
    reclab = property(lambda self:" ".join([ ("[%s]" if i == self.irec else "%s") % sq for i,sq in enumerate(self.sqs)]))

    @classmethod
    def reclab_(cls, seq0, irec ):
        return " ".join([ ("[%s]" if i == irec else "%s") % sq for i,sq in enumerate(seq0.split())])

    @classmethod
    def reclabs_(cls, seq0):
        """
        :param seq0: sequence label such as 'TO RE RE RE RE BT BT BT SC BR BR BR BR BR BR BR'
        :return reclabel list: with each irec highlighted sequentially 

        Example::

            In [9]: rl[:5]
            Out[9]: 
            ['[TO] BT BT BT BT SA',
             'TO [BT] BT BT BT SA',
             'TO BT [BT] BT BT SA',
             'TO BT BT [BT] BT SA',
             'TO BT BT BT [BT] SA']


        reclabs are strings of up to 16*3+2 = 50 chars
        """
        sqs = seq0.split()
        nsqs = len(sqs)

        sqlab = np.zeros( (16), dtype="|S4" )
        sqlab[:nsqs] = sqs 

        rls = np.tile(sqlab, nsqs).reshape(nsqs,-1)
        for ir in range(nsqs):
            rls[ir,ir] = b_("[") + rls[ir,ir] + b_("]")  
        pass
        rlabs = map(lambda ir:" ".join(lmap(d_, rls[ir][rls[ir] != b_("")])), range(nsqs))  
        return rlabs  


    def qsub(self):
        subs = []
        for q in list(self.QWNS):
            subs.append(Ctx(self, qwn=q))
        return subs
    
    def _get_suptitle(self):
        return " %s/%s %s " % (self.det, self.tag, self.reclab )
    suptitle = property(_get_suptitle)

    @classmethod
    def srec_(cls, irec):
        """
        :param irec: decimal int
        :return srec: single char hexint 
        """
        srec = "%x" % irec  
        assert len(srec) == 1, (irec, srec, "expecting single char hexint string")
        return srec 



    @classmethod
    def base(cls):
        return os.path.expandvars(cls.BASE)

    @classmethod
    def tagdir_(cls, ctx):
        return os.path.expandvars(os.path.join(cls.BASE,ctx["det"],ctx["tag"]))

    @classmethod
    def irec_(cls, srec):
        """
        :param srec: one or more single char hexint
        :return irec: one or more ints 
        """
        return [int(c,16) for c in list(srec)]

    def _get_dir(self):
        qctx = self.qctx
        tagd = self.tagdir_(self)
        return os.path.join(tagd,qctx)
    dir = property(_get_dir)

    def path(self, name):
        return os.path.join(self.dir, name)

    @classmethod
    def debase_(cls, dir_):
        """
        :param dir_:
        :return pctx, adir: path string context, absolute dir
        """
        base = cls.base()
        if dir_.startswith(base):
            pctx = dir_[len(base)+1:] 
        else:
            pctx = dir_
        pass
        adir = os.path.join(base, pctx)
        return pctx, adir

    @classmethod
    def det_tag_seq0s_(cls, ctxs):
        """
        :param ctxs: flat list of ctx
        :return seq0s: unique list of seq0 
        """
        dets = list(set(map(lambda ctx:ctx.get("det", None),ctxs)))
        tags = list(set(map(lambda ctx:ctx.get("tag", None),ctxs)))
        seq0s = list(set(map(lambda ctx:ctx.get("seq0", None),ctxs)))

        assert len(dets) == 1, (dets, "multiple dets in ctx list not supported")
        assert len(tags) == 1, (tags, "multiple tags in ctx list not supported")
        assert len(seq0s) >= 1, (seq0s, "unexpected seq0 in ctx list ")

        return dets[0], tags[0], seq0s 

    @classmethod
    def filter_ctx_(cls, ctxs, seq0):
        """
        :param ctxs: flat list of ctx dicts
        :param seq0: string 
        :return fctx: filtered list ctx dicts 
        """
        return filter(lambda ctx:ctx.get("seq0",None) == seq0, ctxs)

    @classmethod
    def reclab_(cls, ctxs):
        """
        :param ctxs: flat list of ctx dicts
        :return reclab: label like 'TO BT BT BT BT [SC] SA'
        """
        rls = list(set(map(lambda ctx:ctx.reclab, ctxs)))
        n_rls = len(rls)
        if n_rls > 1:
            log.fatal("n_rls %d " % n_rls )
            for ictx, ctx in enumerate(ctxs):
                log.fatal("ictx %d  ctx %s " % (ictx, repr(ctx)))
            pass
        pass
        assert n_rls == 1, rls
        return rls[0]

    @classmethod
    def dir2ctx_(cls, dir_, **kwa):
        """
        :param dir_:
        :return list of ctx:

        Expect absolute or relative directory paths such as::

            dir_ = "/tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X"
            dir_ = "concentric/1/TO_BT_BT_BT_BT_SA/0/X"

        Last element "X" represents the quantity or quantities, with one or more of "XYZTABCR".

        Penultimate element "0" represents the irec index within the sequence, with 
        one or more single chars from "0123456789abcdef". For example "0" points to 
        the "TO" step for seq0 of "TO_BT_BT_BT_BT_SA".

        """
        pctx, adir = cls.debase_(dir_)
        return cls.pctx2ctx_(pctx, **kwa)


    @classmethod
    def reclab2ctx_(cls, arg, **kwa):
        """
        Requires reclab of below form, with a single point highlighted.
        Applies reclab into Ctx conversion, eg::

             "[TO] BT BT BT BT SA"                             ->  Ctx(seq0="TO BT BT BT BT SA", irec=0)
             "TO BT BT BT BT DR BT BT BT BT BT BT BT BT [SA]"  ->  Ctx(seq0="TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA", irec=14)

        """
        arg_ = arg.strip() 
        assert arg_.find(" ") > -1, (arg_, "expecting reclab argument with spaces ")

        recs = filter(lambda _:_[1][0] == "[" and _[1][3] == "]", list(enumerate(arg_.split())))
        assert len(recs) == 1
        irec = int(recs[0][0])

        elem = arg_.replace("[","").replace("]","").split()
        seq0 = " ".join(elem)

        return Ctx(seq0=seq0, irec=irec, **kwa)


    @classmethod
    def pctx2ctx_(cls, pctx, **kwa):
        """
        :param pctx:
        :return list of ctx:

        Full pctx has 5 elem::

            pctx = "concentric/1/TO_BT_BT_BT_BT_SA/0/X"
  
        Shorter qctx has 3 elem or 2 elem::

            qctx = "TO_BT_BT_BT_BT_SA/0/X"
            qctx = "TO_BT_BT_BT_BT_SA/0"

        Not using os.listdir for this as want to 
        work more generally prior to persisting and with 
        path contexts that do not directly correspond to file system paths.
        """
        ctxs = []
        e = pctx.split("/")
        ne = len(e)
        if ne == 5:
            ks = 2
            kr = 3
            kq = 4
        elif ne == 3:
            ks = 0
            kr = 1
            kq = 2
        elif ne == 2:
            ks = 0
            kr = 1
            kq = None
        else:
            log.warning("unexpected path context %s " % pctx )
            return []

        if kq is None:
            qwns = cls.QWNS
        else:
            qwns = e[kq]

        log.info("pctx2ctx_ ne %d qwns %s e[kr] %s " % (ne, qwns, e[kr]) )


        for r in e[kr]:
            #ir = str(int(r,16))
            ir = int(r,16)
            for q in qwns:
                ctx = dict(seq0=e[ks],irec=ir,qwn=q)
                if ne == 5:
                    ctx.update({"det":e[0], "tag":e[1]})
                elif ne == 3 or ne == 2:
                    ctx.update(kwa)
                else:
                    pass
                pass
                ctxs.append(Ctx(ctx))
                pass
            pass
        pass
        return ctxs




def test_reclab2ctx_():
     chks = {
                                             "[TO] BT BT BT BT SA":"TO_BT_BT_BT_BT_SA/0/%s" % Ctx.QWNS,
                  "TO BT BT BT BT DR BT BT BT BT BT BT BT [BT] SA":"TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/d/%s" % Ctx.QWNS,
                  "TO BT BT BT BT DR BT BT BT BT BT BT BT BT [SA]":"TO_BT_BT_BT_BT_DR_BT_BT_BT_BT_BT_BT_BT_BT_SA/e/%s" % Ctx.QWNS,
            }

     for k,qctx_x in chks.items():
         ctx = Ctx.reclab2ctx_(k)
         qctx = ctx.qctx
         log.info(" %50s -> %50r -> %s " % (k, ctx, qctx ))
         assert qctx == qctx_x, ( qctx, qctx_x ) 

def test_reclab2ctx_2(ok):
     reclab = "[TO] AB"
     ctx = Ctx.reclab2ctx_(reclab, det=ok.det, tag=ok.tag)
     print(ctx)
 


def test_pctx2ctx_5():

     chks = { 
              "concentric/1/TO_BT_BT_BT_BT_SA/0/X":'TO_BT_BT_BT_BT_SA/0/X' ,
            }

     for k,qctx_x in chks.items():
         ctxs = Ctx.pctx2ctx_(k)
         assert len(ctxs) == 1
         ctx = ctxs[0]
         qctx = ctx.qctx     
         log.info( " %50s -> %r " % (k, qctx))
         assert qctx == qctx_x 


def test_pctx2ctx_2():

     chks = { 
              "TO_BT_BT_BT_BT_SA/0":None ,
            }

     for k,qctx_x in chks.items():
         ctxs = Ctx.pctx2ctx_(k)
         assert len(ctxs) == 8

         for ctx in ctxs:
             log.info( " %r -> %s " % (ctx, ctx.pctx))


def test_reclabs_0():
    """
    This works in py2+py3 but its ugly, because py3 makes it 
    awkward to flip back and forth between numpy bytes and strings. 

    Solution is to avoid the flipping : do everything in numpy bytes 
    and then flip back to python just as the last step.
    """

    #seq0 = "AA BB CC DD EE FF GG HH II JJ KK LL MM NN OO PP"
    seq0 = "AA BB CC DD EE FF GG HH"

    sqs = seq0.split()
    nsqs = len(sqs)

    sqlab = np.zeros( (16), dtype="|S4" )
    sqlab[:nsqs] = sqs 

    rls = np.tile(sqlab, nsqs).reshape(nsqs,-1)
    for ir in range(nsqs):
        rls[ir,ir] = b_("[") + rls[ir,ir] + b_("]")  
    pass

    rlabs = lmap(lambda ir:" ".join(lmap(d_,lfilter(lambda _:_ != b_(""),rls[ir]))), range(nsqs))

    for rlab in rlabs:
        print(rlab)
    pass


def test_reclabs_1():

    seq0 = "AA BB CC DD EE FF GG HH"

    sqs = seq0.split()
    nsqs = len(sqs)

    sqlab = np.zeros( (16), dtype="|S4" )
    sqlab[:nsqs] = sqs 

    rls = np.tile(sqlab, nsqs).reshape(nsqs,-1)
    for ir in range(nsqs):
        rls[ir,ir] = b_("[") + rls[ir,ir] + b_("]")  
    pass

    rlabs = []
    for ir in range(nsqs):
        a = rls[ir]
        l = a[a != b_("")] 
        rlab = " ".join(lmap(d_, l))  
        rlabs.append(rlab)
    pass

    for rlab in rlabs:
        print(rlab)
    pass


def test_reclabs_2():
    seq0 = "AA BB CC DD EE FF GG HH"

    sqs = seq0.split()
    nsqs = len(sqs)

    sqlab = np.zeros( (16), dtype="|S4" )
    sqlab[:nsqs] = sqs 

    rls = np.tile(sqlab, nsqs).reshape(nsqs,-1)
    for ir in range(nsqs):
        rls[ir,ir] = b_("[") + rls[ir,ir] + b_("]")  
    pass
    rlabs = map(lambda ir:" ".join(lmap(d_, rls[ir][rls[ir] != b_("")])), range(nsqs))  

    for rlab in rlabs:
        print(rlab)
    pass


    
if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()

    #test_reclab2ctx_()
    #test_reclab2ctx_2(ok)
    #test_pctx2ctx_5()
    #test_pctx2ctx_2()


    #test_reclabs_0() 
    #test_reclabs_1() 





