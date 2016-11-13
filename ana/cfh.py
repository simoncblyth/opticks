#!/usr/bin/env python
"""

* TODO: complete the decoupling of chi2 distrib tables creation and plotting

* chi2 tables are made with make_rst_table.py but the 
  creation of stat.npy currently tied into plotting machinery 
  which limits usage to small numbers of distribs


CFH random access for debugging::

    delta:~ blyth$ ipython -i $(which cfh.py) -- /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    /Users/blyth/opticks/ana/cfh.py /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X
    ['/tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X']
    [2016-11-12 18:42:19,579] p4851 {/Users/blyth/opticks/ana/cfh.py:199} INFO - CFH.load from /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X 

    In [1]: h
    Out[1]: X[0]

    In [2]: h.bins
    Out[2]: array([-76.3726, -61.0981, -45.8235, -30.549 , -15.2745,   0.    ,  15.2745,  30.549 ,  45.8235,  61.0981,  76.3726])


"""
import os, sys, json, logging, numpy as np
from opticks.ana.base import opticks_main, json_
from opticks.ana.nbase import chi2
log = logging.getLogger(__name__)


class CFH(object):
    """
    Persistable comparison histograms and chi2
    The members are numpy arrays and a single ctx dict
    allowing simple load/save.
    """
    NAMES = "bins ahis bhis chi2".split()
    BASE = "$TMP/CFH"

    @classmethod
    def base(cls):
        return os.path.expandvars(cls.BASE)

    @classmethod
    def dir_(cls, ctx):
        seq0 = ctx["seq0"]
        if seq0 is None:
            log.fatal("CFH histograms requires single line selections")
            return None
        pass
        return os.path.expandvars(os.path.join(cls.BASE,ctx["det"],ctx["tag"],seq0,str(ctx["irec"]),ctx["qwn"]))


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
    def pctx2ctx_(cls, pctx):
        """
        :param pctx:
        :return list of ctx:

        Not using os.listdir for this as want to 
        work more generally prior to persisting and with 
        path contexts that do not directly correspond to file system paths.
        """
        ctxs = []
        e = pctx.split("/")
        ne = len(e)
        if ne == 5:
            for r in e[3]:
                ir = str(int(r,16))
                for q in e[4]:
                    ctx = dict(det=e[0],tag=e[1],seq0=e[2],irec=ir,qwn=q)
                    ctxs.append(ctx)
                pass
            pass
        else:
            log.warning("unexpected path context %s " % pctx )
            return []
        pass
        return ctxs



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
    def dir2ctx_(cls, dir_):
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
        return cls.pctx2ctx_(pctx)

    def dir(self):
        return self.dir_(self.ctx)

    seq0 = property(lambda self:self.ctx.get("seq0", None))

    def _get_suptitle(self):
        return self.dir()
    suptitle = property(_get_suptitle)

    @classmethod
    def path_(cls, ctx, name):
        dir_ = cls.dir_(ctx)
        return os.path.join(dir_, name)

    @classmethod
    def load_(cls, ctx):
         if type(ctx) is list:
             return map(lambda _:cls.load_(_), ctx)
         elif type(ctx) is dict:
             h = CFH(ctx)  
             h.load()
             return h 
         else:
             log.warning("CFH.load_ unexpected ctx %s " % repr(ctx))
         return None

    def path(self, name):
        return self.path_(self.ctx, name)

    def __init__(self, ctx={}):
        if type(ctx) is str:
            ctxs = self.dir2ctx_(ctx)
            assert len(ctxs) == 1, "expect only a single ctx"
            ctx = ctxs[0]
        pass
        self.ctx = ctx 

        # transients, not persisted
        self._log = False

    def __call__(self, bn, av, bv, lab, cut=30):
        self.bins = bn
        self.ahis,_ = np.histogram(av, bins=bn)
        self.bhis,_ = np.histogram(bv, bins=bn)
        c2, c2n, c2c = chi2(self.ahis.astype(np.float32), self.bhis.astype(np.float32), cut=cut)
        self.chi2 = c2

        meta = {}
        meta['cut'] = cut  
        meta['c2n'] = c2n  
        meta['c2c'] = c2c 
        meta['la'] = lab[0] 
        meta['lb'] = lab[1] 

        meta['c2_ymax'] = "10"
        meta['logyfac'] = "3."
        meta['linyfac'] = "1.3"

        self.ctx.update(meta)

    def _get_ndf(self):
        ndf = max(self.c2n - 1, 1)
        return ndf 
    ndf = property(_get_ndf)

    def _get_c2p(self):
        ndf = self.ndf 
        c2p = self.chi2.sum()/ndf
        return c2p 
    c2p = property(_get_c2p)
    
    def _get_c2label(self):   
        return "chi2/ndf %4.2f [%d]" % (self.c2p, self.ndf)
    c2label = property(_get_c2label)


    def _set_log(self, log_=True):
        self._log = log_
    def _get_log(self):
        return self._log
    log = property(_get_log, _set_log) 

    def _get_ylim(self):
        ymin = 1 if self.log else 0 
        yfac = self.logyfac if self.log else self.linyfac
        ymax = max(self.ahis.max(), self.bhis.max()) 
        return [ymin,ymax*yfac]
    ylim = property(_get_ylim)


    def _get_ctxstr(self, name, fallback="?"):
        return str(self.ctx.get(name,fallback))

    def _get_ctxfloat(self, name, fallback="0"):
        return float(self.ctx.get(name,fallback))

    la = property(lambda self:self._get_ctxstr("la"))
    lb = property(lambda self:self._get_ctxstr("lb"))

    c2_ymax = property(lambda self:self._get_ctxfloat("c2_ymax"))
    logyfac = property(lambda self:self._get_ctxfloat("logyfac"))
    linyfac = property(lambda self:self._get_ctxfloat("linyfac"))
    c2n = property(lambda self:self._get_ctxfloat("c2n"))
    c2c = property(lambda self:self._get_ctxfloat("c2c"))
    cut = property(lambda self:self._get_ctxfloat("cut"))

    def __repr__(self):
        return "%s[%s]" % (self.ctx['qwn'],self.ctx['irec'])

    def ctxpath(self):
        return self.path("ctx.json") 

    def paths(self):
        return [self.ctxpath()] + map(lambda name:self.path(name+".npy"), self.NAMES)

    def exists(self):
        if self.seq0 is None:
            log.warning("CFH.exists can only be used with single line selections")
            return False 

        paths = self.paths()
        a = np.zeros(len(paths), dtype=np.bool)
        for i,path in enumerate(paths):
            a[i] = os.path.exists(path)
        return np.all(a[i] == True)

    def save(self):
        if self.seq0 is None:
            log.warning("CFH.save can only be used with single line selections")
            return  

        dir_ = self.dir_(self.ctx)
        log.info("CFH.save to %s " % dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        log.info("saving to %s " % dir_)
        json.dump(self.ctx, file(self.ctxpath(),"w") )
        for name in self.NAMES:
            np.save(self.path(name+".npy"), getattr(self, name))
        pass
 

    def load(self):
        if self.seq0 is None:
            log.warning("CFH.load can only be used with single line selections")
            return  

        dir_ = self.dir_(self.ctx)
        assert os.path.exists(dir_)
        log.info("CFH.load from %s " % dir_)
        js = json_(self.ctxpath())
        k = map(str, js.keys())
        v = map(str, js.values())
        self.ctx = dict(zip(k,v))
        for name in self.NAMES:
            setattr(self, name, np.load(self.path(name+".npy")))
        pass



def test_load():
    ctx = {'det':"concentric", 'tag':"1", 'qwn':"X", 'irec':"5", 'seq':"TO_BT_BT_BT_BT_DR_SA" }
    h = CFH_.load(ctx)



if __name__ == '__main__':
    ok = opticks_main(tag="1", src="torch", det="concentric")

    import matplotlib.pyplot as plt
    plt.rcParams["figure.max_open_warning"] = 200    # default is 20
    plt.ion()

    from opticks.ana.ab import AB
    from opticks.ana.cfplot import one_cfplot, qwns_plot 
    print ok.nargs

    if ok.rehist:
        ab = AB(ok)
    else:
        ab = None

    if len(ok.nargs) > 0:
        pctx = ok.nargs[0]

        ctxs = CFH.dir2ctx_(pctx)
        det, tag, seq0s = CFH.det_tag_seq0s_(ctxs)

        for seq0 in seq0s:
            ctxs = CFH.filter_ctx_(ctxs, seq0)  
            suptitle = " %s %s %s " % (det, tag, seq0)
            if ok.rehist:
                hh = ab.rhist_(ctxs)
            else:
                hh = CFH.load_(ctxs)
            pass
            if len(hh) == 1:
                one_cfplot(hh[0])
            else:
                qwns_plot(hh, suptitle  )
            pass
        pass
    pass



