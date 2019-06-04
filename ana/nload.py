#!/usr/bin/env python
"""
nload.py
==========

::

   cd /tmp
   OPTICKS_EVENT_BASE=tboolean-box nload.py

"""
import os, sys, logging, datetime
log = logging.getLogger(__name__)

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
pass

import numpy as np
from opticks.ana.base import ini_


def stmp_(st, fmt="%Y%m%d-%H%M"): 
    return datetime.datetime.fromtimestamp(st.st_ctime).strftime(fmt)

def stamp_(path, fmt="%Y%m%d-%H%M"):
    try:
        st = os.stat(path)
    except OSError:
        return "FILE-DOES-NOT-EXIST"
    return stmp_(st, fmt=fmt)

def x_(_):
    p = os.path.expandvars(_)
    st = stamp_(p)
    log.info( " %s -> %s (%s) " % (_, p, st))
    return p  

txt_ = lambda _:np.loadtxt(StringIO(_))


def np_load(base,sub=None,rel=None):
    """
    Loads np array, returning None if non-existing 
    """
    path_ = os.path.join(*filter(None,[base,sub,rel]))    
    path = os.path.expandvars(path_)

    if os.path.exists(path):
        a = np.load(path)
    else: 
        log.warning("np_load path_:%s path:%s DOES NOT EXIST " % ( path_, path ))
        a = None 
    pass
    return a


DEFAULT_BASE = "$OPTICKS_EVENT_BASE/$0/evt"
DEFAULT_DIR_TEMPLATE = DEFAULT_BASE + "/$1/$2/$3"  ## cf C++  brap- BOpticksEvent

def tagdir_(det, typ, tag, pfx=".", layout=2):
    """
    layout version 1 (which is still used for source gensteps) does
    not have the tag in the directory 

    OPTICKS_EVENT_BASE
        in direct running this is the geocache directory, with "$0" pfx
        being the name of the executable or "source" for the primary (geocache creating)
        executable

        for test running from legacy basis geometry this is for example /tmp
        with pfx being the name of the test eg "tboolean-box"      

    """
    log.debug("tagdir_ det %s typ %s tag %s layout %s DEFAULT_DIR_TEMPLATE %s " % (det,typ,tag,layout, DEFAULT_DIR_TEMPLATE))
    log.debug("tagdir_ type(tag) %s " % type(tag)) 

    if layout == 1: 
        utag = "."
        tmpl = DEFAULT_DIR_TEMPLATE
        tmpl = tmpl.replace("$0", pfx)
        tmpl = tmpl.replace("$1", det)
        tmpl = tmpl.replace("$2", typ)
        tmpl = tmpl.replace("$3",utag)
    elif layout == 2:
        tmpl = DEFAULT_DIR_TEMPLATE
        tmpl = tmpl.replace("$0", pfx)
        tmpl = tmpl.replace("$1", det)
        tmpl = tmpl.replace("$2", typ)
        if tag is not None:  
            tmpl = tmpl.replace("$3", str(tag))
        else:
            tmpl = tmpl.replace("$3", "/")
        pass
    else:
        assert 0, "bad layout"

    log.debug("tagdir_ tmpl %s " % tmpl )
    xdir = os.path.expandvars(tmpl)
    while xdir.endswith("/"):xdir = xdir[:-1]

    if not os.path.exists(xdir):
        log.error("NON EXISTING tagdir : %s  expanded from %s " % (xdir, DEFAULT_DIR_TEMPLATE))
        log.error("As relative paths are used with test geometry running, subsequent scripts or executables that intend to reuse data should be invoked from the same directory." )
        assert 0, (xdir, tmpl, DEFAULT_DIR_TEMPLATE)
    pass
    return xdir


def typdirs_(evtdir=None):
    """
    :return typdirs: list of absolute paths to type dirs 
    """
    if evtdir is None: 
        evtdir = os.path.expandvars("/tmp/$USER/opticks/evt")
    pass
    dets = os.listdir(evtdir)
    typdirs = []
    for det in filter(lambda det:os.path.isdir(os.path.join(evtdir,det)),os.listdir(evtdir)):
        detdir = os.path.join(evtdir, det)
        for typ in os.listdir(detdir): 
            typdir = os.path.join(detdir, typ)
            print("typdir : ", typdir)
            typdirs.append(typdir)
        pass
    pass
    return typdirs

def path_(typ, tag, det="dayabay", name=None, pfx="source" ):
    """
    :param typ:
    :param tag:
    :param det:
    :param name:
    :param layout:

    *layout 1*
         typ is of form oxtorch with the 
         constituent and source combined and the tag is in the
         filename stem.

    *layout 2*
         tag is in the directory and the filename stem 
         is the constituent eg ox  


    Signal use of layout 2 by specifying the "ox" "rx" "idom" etc
    in the name rather than on the "typ"
    """
    if name is None:
        layout = 1
    else:
        layout = 2
    pass
    tagdir = tagdir_(det, typ, tag, layout=layout, pfx=pfx )
    if layout == 1:
        tmpl = os.path.join(tagdir, "%s.npy" % tag) 
    elif layout == 2:
        tmpl = os.path.join(tagdir, name) 
    else:
        assert 0, "bad layout"
    pass
    return tmpl 


def tpaths_(typ, tag, det="dayabay", name=None, pfx="source"):
    """
    :return tnams: paths of files named *name* that are contained in directories within the tagdir 
                   typically these are time stamped dirs 
    """
    assert name is not None 

    tagdir = tagdir_(det, typ, tag, pfx=pfx)
   
    if os.path.isdir(tagdir): 
        names = os.listdir(tagdir)
    else:
        log.warning("tpaths_ tagdir %s does not exist" % tagdir)
        names = []
    pass

    tdirs = filter( os.path.isdir, map(lambda name:os.path.join(tagdir, name), names))
    tdirs = sorted(tdirs, reverse=True)  # paths of dirs within tagdir
    tnams = filter( os.path.exists, map(lambda tdir:os.path.join(tdir, name), tdirs)) 
    # paths of named files 
    return tnams


def gspath_(typ, tag, det, gsbase=None):
    if gsbase is None:
        gsbase= os.path.expandvars("$LOCAL_BASE/opticks/opticksdata/gensteps")
    gspath = os.path.join(gsbase, det, typ, "%s.npy" % tag)
    return gspath 

class A(np.ndarray):
    @classmethod
    def load_(cls, stem, typ, tag, det="dayabay", pfx="source", dbg=False, optional=False):
        """
        :param stem: gs,ox,ht,rs,so,ph,fdom,idom
        :param typ: natural
        :param tag: 1,-1
        :param det: g4live 
        :param pfx: source for 1st executable, the name of the executable for subsequent ones eg OKG4Test 
        :param dbg: 
        :param optional: 
        """
        if dbg:
            print("stem:%s typ:%s tag:%s det:%s pfx:%s dbg:%s optional:%s" % (stem, typ, tag, det, pfx, dbg, optional))
        pass

        if stem == "gensteps":
            path = gspath_(typ, tag, det)
            assert 0, "suspect these paths are now wrong"
        else:
            path = path_(typ,tag, det, name="%s.npy" % stem, pfx=pfx)
        pass

        a = None
        missing = False  
        
        if os.path.exists(path):
            log.debug("loading %s " % path )
            if dbg: 
                os.system("ls -l %s " % path)
            arr = np.load(path)
        else:
            if not optional: 
                raise IOError("cannot load %s " % path)
            arr = np.zeros(())
            missing = True

        a = arr.view(cls)
        a.path = path 
        a.typ = typ
        a.tag = tag
        a.det = det 
        a.pfx = pfx
        a.stamp = stamp_(path)
        a.missing = missing

        return a

    def __repr__(self):
        if hasattr(self, 'typ'):
            return "A(%s,%s,%s)%s\n%s" % (self.typ, self.tag, self.det, getattr(self,'desc','-'),np.ndarray.__repr__(self))
        pass
        return "A()sliced\n%s" % (np.ndarray.__repr__(self))
 
    def derivative_path(self, postfix="track"):
        tag = "%s_%s" % (self.tag, postfix)
        return path_(self.typ, tag, self.det )

    def derivative_save(self, drv, postfix="track"): 
        path = self.derivative_path(postfix)
        if os.path.exists(path):
            log.warning("derivative of %s at path %s exists already, delete and rerun to update" % (repr(self),path) )
        else:
            log.info("saving derivative of %s to %s " % (repr(self), path ))
            np.save( path, drv )    



class I(dict):
    @classmethod
    def loadpath_(cls, path, dbg=False):
        if os.path.exists(path):
            log.debug("loading %s " % path )
            if dbg: 
                os.system("ls -l %s " % path)
            pass
            d = ini_(path)
        else:
            #log.warning("cannot load %s " % path)
            raise IOError("cannot load %s " % path)
            d = {}
        return d  

    @classmethod
    def load_(cls, typ, tag, det="dayabay", pfx="source", name="DeltaTime.ini", dbg=False):
        path = path_(typ, tag, det, name=name, pfx=pfx )
        i = cls(path, typ=typ, tag=tag, det=det, name=name)
        return i

    def __init__(self, path, typ=None, tag=None, det=None, name=None, dbg=False):
        d = self.loadpath_(path,dbg=dbg)
        dict.__init__(self, d)
        self.path = path
        self.stamp = stamp_(path)
        self.fold = os.path.basename(os.path.dirname(path))
        self.typ = typ
        self.tag = tag
        self.det = det
        self.name = name

    def __repr__(self):
        return "I %5s %10s %10s %10s %10s " % (self.tag, self.typ, self.det, self.name, self.fold )


class II(list):
    """  
    List of I instances holding ini file dicts, providing a history of 
    event metadata
    """
    @classmethod
    def load_(cls, typ, tag, det="dayabay", pfx="source", name="DeltaTime.ini", dbg=False):
        tpaths = tpaths_(typ, tag, det, pfx=pfx, name=name)
        ii = map(lambda path:I(path, typ=typ, tag=tag, det=det, name=name, dbg=dbg), tpaths) 
        return cls(ii)

    def __init__(self, ii):
        list.__init__(self, ii)   

    def __getitem__(self, k):
        return map(lambda d:d.get(k, None), self) 

    def folds(self):
        return map(lambda i:i.fold, self) 

    def __repr__(self):
        return "\n".join(map(repr, self))



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    #ok = opticks_main(src="torch", tag="10", det="PmtInBox")
    ok = opticks_main()

    try:
        i = I.load_(typ=ok.src, tag=ok.tag, det=ok.det, pfx=ok.pfx, name="DeltaTime.ini")
    except IOError as err:
        log.fatal(err)
        sys.exit(ok.mrc)

    log.info(" loaded i %s %s " % (i.path, i.stamp))
    print("\n".join([" %20s : %s " % (k,v) for k,v in i.items()]))


    #a = A.load_("ph", "torch","5", "rainbow")
    #i = I.load_("torch","5", "rainbow", name="t_delta.ini")

    #ii = II.load_("torch","5", "rainbow", name="t_delta.ini")
    #iprp = map(float, filter(None,ii['propagate']) )

    #jj = II.load_("torch","-5", "rainbow", name="t_delta.ini")
    #jprp = map(float, filter(None,jj['propagate']) )


 
