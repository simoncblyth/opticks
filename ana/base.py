#!/usr/bin/env python
"""
Mostly Non-numpy basics, just numpy configuration
"""

import numpy as np
import os, logging, json, ctypes, subprocess, argparse, sys, datetime, re
from OpticksQuery import OpticksQuery 
from opticks.ana.enum import Enum 

log = logging.getLogger(__name__) 


try:
    cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
    ffs_ = lambda _:cpp.ffs(_)
except OSError:
    pass

IDPATH = os.path.expandvars("$IDPATH")
idp_ = lambda _:"%s/%s" % (IDPATH,_) 
uidp_ = lambda _:_.replace(IDPATH,"$IDPATH")


def translate_xml_identifier_(name):
    return name.replace("__","/").replace("--","#").replace("..",":") 

class Buf(np.ndarray): pass

splitlines_ = lambda txtpath:file(txtpath).read().splitlines()

def now_(fmt="%Y%m%d-%H%M"):
    return datetime.datetime.now().strftime(fmt)

def stamp_(path, fmt="%Y%m%d-%H%M"): 
   if path is None:
       return None
   elif not os.path.exists(path):
       return None
   else:
       return datetime.datetime.fromtimestamp(os.stat(path).st_ctime).strftime(fmt)
   pass


def is_integer_string(s):
    try:
        int(s)
        iis = True
    except ValueError:
        iis = False
    pass
    return iis

def list_integer_subdirs(base):
    """
    return list of subdirs beneath base with names that are lexically integers, sorted as integers  
    """
    return list(sorted(map(int,filter(is_integer_string,os.listdir(os.path.expandvars(base))))))

def dump_extras_meta(base, name="meta.json", fmt=" %(idx)5s : %(height)6s : %(lvname)-40s : %(soname)-40s : %(err)s "):
    """
    Tabulate content of meta.json files from subdirs with integer names
    """
    idxs = list_integer_subdirs(base)
    assert idxs == range(len(idxs))

    log.info("dump_extras_meta base:%s xbase:%s " % (base,expand_(base)))

    keys = re.compile("\((\w*)\)").findall(fmt)
    print fmt % dict(zip(keys,keys))

    for idx in idxs:
        meta = json_load_(os.path.join(base,str(idx),name))
        meta['idx'] = idx
        meta['err'] = meta.get('err',"-")
        print fmt % meta
    pass




def makedirs_(path):
    pdir = os.path.dirname(path)
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    pass
    return path 

expand_ = lambda path:os.path.expandvars(os.path.expanduser(path))
json_load_ = lambda path:json.load(file(expand_(path)))
json_save_ = lambda path, d:json.dump(d, file(makedirs_(expand_(path)),"w"))
json_save_pretty_ = lambda path, d:json.dump(d, file(makedirs_(expand_(path)),"w"), sort_keys=True, indent=4, separators=(',', ': '))



def manual_mixin( dst, src ):
    """ 
    Add all methods from the src class to the destination class

    :param dst: destination class
    :param src: source class
    """
    for k,fn in src.__dict__.items():
        if k.startswith("_"): continue
        setattr(dst, k, fn ) 
    pass


def _dirname(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path

def _opticks_idfold(idpath):
    idfold = _dirname(idpath,1)
    log.debug("_opticks_idfold idpath %s -> idfold %s " % (idpath, idfold))
    return idfold

def _opticks_idfilename(idpath):
    name = os.path.basename(idpath)
    elem = name.split(".")
    assert len(elem) == 3
    idfilename = "%s.%s" % (elem[0],elem[2])
    log.debug("_opticks_idfilename idpath %s -> idfilename %s " % (idpath, idfilename))
    return idfilename

def _opticks_daepath(idpath):
    idfilename = _opticks_idfilename(idpath)
    idfold = _opticks_idfold(idpath)
    return os.path.join(idfold, idfilename)

def _opticks_gdmlpath(idpath):
    idfilename_dae = _opticks_idfilename(idpath)
    base,ext = idfilename_dae.split(".")
    idfilename_gdml = "%s.gdml" % base 
    idfold = _opticks_idfold(idpath)
    return os.path.join(idfold, idfilename_gdml)

def _opticks_gltfpath(idpath):
    idfilename_dae = _opticks_idfilename(idpath)
    base,ext = idfilename_dae.split(".")
    idfilename_gltf = "%s.gltf" % base 
    idfold = _opticks_idfold(idpath)
    return os.path.join(idfold, idfilename_gltf)


def _opticks_install_prefix(idpath):
    prefix = _dirname(idpath,4)
    log.debug("_opticks_install_prefix idpath %s -> prefix %s " % (idpath, prefix))
    return prefix 

def _opticks_data_dir(idpath):
    datadir = _dirname(idpath,3)
    log.debug("_opticks_datadir_dir idpath %s -> datadir %s " % (idpath, datadir))
    return datadir 

def _opticks_export_dir(idpath):
    xdir = _dirname(idpath,2)
    log.debug("_opticks_exportdir_dir idpath %s -> xdir %s " % (idpath, xdir))
    return xdir 

def _opticks_install_cache(idpath):
    prefix = _opticks_install_prefix(idpath) 
    path = os.path.join(prefix, "installcache") 
    log.debug("_opticks_install_cache idpath %s -> datadir %s " % (idpath, path))
    return path 

def _opticks_detector(idpath):
    ddir = idpath.split("/")[-2]
    dbeg = ddir.split("_")[0]
    if dbeg in ["DayaBay","LingAo","Far"]:
        detector =  "DayaBay"
    else:
        detector = dbeg 
    pass
    log.debug("_opticks_detector idpath %s -> detector %s " % (idpath, detector))
    return detector 

def _opticks_detector_dir(idpath):
    detector = _opticks_detector(idpath)
    xdir = _opticks_export_dir(idpath)
    return os.path.join(xdir, detector)


def _subprocess_output(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.communicate()

def _opticks_default_idpath_from_exe(exe="OpticksIDPATH"):
    """
    This provides a way to discern the IDPATH by running an 
    Opticks executable.  This works fine for the default geometry with 
    no arguments picking alternate geometries.  
    
    To make this work for non default geometries would have to 
    somehow save the opticks geometry selection arguments, given 
    this might as way stay simple and require the IDPATH envvar 
    as an input to analysis.
    """ 
    stdout, stderr = _subprocess_output([exe]) 
    idpath = stderr.strip()
    return idpath  

def _opticks_event_base():
    return os.path.expandvars("/tmp/$USER/opticks") 
def _opticks_tmp():
    return os.path.expandvars("/tmp/$USER/opticks") 

def _opticks_env(st="OPTICKS_ IDPATH"):
    return filter(lambda _:_[0].startswith(st.split()), os.environ.items())

def _opticks_query():
    return ""

class OpticksEnv(object):
    def __init__(self):
        self.ext = {}
        self.env = {}

        if IDPATH == "$IDPATH":
            print "ana/base.py:OpticksEnv missing IDPATH envvar [%s] " % IDPATH
            sys.exit(1)  

        if not os.path.isdir(IDPATH): 
            print "ana/base.py:OpticksEnv warning IDPATH directory does not exist [%s] " % IDPATH
        pass  
 
        self.setdefault("OPTICKS_IDFOLD",          _opticks_idfold(IDPATH))
        self.setdefault("OPTICKS_IDFILENAME",      _opticks_idfilename(IDPATH))
        self.setdefault("OPTICKS_DAEPATH",         _opticks_daepath(IDPATH))
        self.setdefault("OPTICKS_GDMLPATH",        _opticks_gdmlpath(IDPATH))
        self.setdefault("OPTICKS_GLTFPATH",        _opticks_gltfpath(IDPATH))
        self.setdefault("OPTICKS_DATA_DIR",        _opticks_data_dir(IDPATH))
        self.setdefault("OPTICKS_EXPORT_DIR",      _opticks_export_dir(IDPATH))
        self.setdefault("OPTICKS_INSTALL_PREFIX",  _opticks_install_prefix(IDPATH))
        self.setdefault("OPTICKS_INSTALL_CACHE",   _opticks_install_cache(IDPATH))
        self.setdefault("OPTICKS_DETECTOR",        _opticks_detector(IDPATH))
        self.setdefault("OPTICKS_DETECTOR_DIR",    _opticks_detector_dir(IDPATH))
        self.setdefault("OPTICKS_EVENT_BASE",      _opticks_event_base())
        self.setdefault("OPTICKS_QUERY",           _opticks_query())
        self.setdefault("TMP",                     _opticks_tmp())

    def bash_export(self, path="$TMP/opticks_env.bash"):
        lines = []
        for k,v in self.env.items():
            line = "export %s=%s " % (k,v)
            lines.append(line)    

        path = os.path.expandvars(path) 

        dir_ = os.path.dirname(path)
        if not os.path.isdir(dir_):
             os.makedirs(dir_)

        #print "writing opticks environment to %s " % path 
        open(path,"w").write("\n".join(lines)) 


    def setdefault(self, k, v):
        self.env[k] = v
        if k in os.environ:
            self.ext[k] = os.environ[k]
        else: 
            os.environ[k] = v 

    def dump(self, st=("OPTICKS","IDPATH")):
        for k,v in os.environ.items():
            if k.startswith(st):
                if k in self.ext:
                     msg = "(ext)"
                else:
                     msg = "    "
                pass
                log.info(" %5s%30s : %s " % (msg,k,v))


def opticks_environment(dump=False):
   env = OpticksEnv()
   if dump:
       env.dump()
   env.bash_export() 


def opticks_main(**kwa):
    args = opticks_args(**kwa)
    opticks_environment()
    np.set_printoptions(suppress=True, precision=4, linewidth=200)
    return args


def isIPython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True


class OK(argparse.Namespace):
    pass
    #ipython = property(lambda self:sys.argv[0].endswith("ipython"))
    ipython = isIPython()
    brief = property(lambda self:"tag %s src %s det %s c2max %s ipython %s " % (self.utag,self.src,self.det, self.c2max, self.ipython))

    def _get_ctx(self):
        return dict(tag=self.tag, utag=self.utag, src=self.src, det=self.det)
    ctx = property(_get_ctx)


def opticks_args(**kwa):

    oad_key = "OPTICKS_ANA_DEFAULTS"
    oad = os.environ.get(oad_key,"det=dayabay,src=torch,tag=1")
    defaults = dict(map(lambda ekv:ekv.split("="), oad.split(","))) 
    log.info("envvar %s -> defaults %s " % (oad_key, repr(defaults)))

    det = kwa.get("det", defaults["det"])
    src = kwa.get("src", defaults["src"])
    tag = kwa.get("tag", defaults["tag"])

    llv = kwa.get("loglevel", "info")
    mrc = kwa.get("mrc", 101)
    doc = kwa.get("doc", None)
    tagoffset = kwa.get("tagoffset", 0)
    multievent = kwa.get("multievent", 1)
    stag = kwa.get("stag", None)
    ptag = kwa.get("ptag", None)
    show = kwa.get("show", True)
    plot = kwa.get("plot", True)
    terse = kwa.get("terse", False)
    mat = kwa.get("mat", "GdDopedLS")
    sli = kwa.get("sli", "::1")
    sel = kwa.get("sel", "0:5:1")
    qwn = kwa.get("qwn", "XYZT,ABCW")
    c2max = kwa.get("c2max", 2.0)
    pfxseqhis = kwa.get("pfxseqhis", "")
    pfxseqmat = kwa.get("pfxseqmat", "")
    dbgseqhis = kwa.get("dbgseqhis", "0")
    dbgseqmat = kwa.get("dbgseqmat", "0")
    dbgmskhis = kwa.get("dbgmskhis", "0")
    dbgmskmat = kwa.get("dbgmskmat", "0")
    smry = kwa.get("smry", True)
    dbgzero = kwa.get("dbgzero", False)
    lmx = kwa.get("lmx", 20)
    cmx = kwa.get("cmx", 0)
    prohis = kwa.get("prohis", False)
    promat = kwa.get("promat", False)
    rehist = kwa.get("rehist", False)
    chi2sel = kwa.get("chi2sel", False)
    chi2selcut = kwa.get("chi2selcut", 1.1)
    statcut = kwa.get("statcut", 1000)
    nointerpol = kwa.get("nointerpol", False)
    figsize = kwa.get("figsize", "18,10.2" )
    size = kwa.get("size", "1920,1080,1" )
    yes = kwa.get("yes", False )
    gltfsave = kwa.get("gltfsave", False )
    lvnlist = kwa.get("lvnlist", "" )

    addpath = kwa.get("addpath", "$LOCAL_BASE/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay.xml" )
    apmtddpath = kwa.get("apmtddpath", "$LOCAL_BASE/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/hemi-pmt.xml" )
    apmtpathtmpl = kwa.get("apmtpathtmpl", "$OPTICKS_INSTALL_PREFIX/opticksdata/export/DayaBay/GPmt/%(apmtidx)s/GPmt.npy" )
    apmtidx = kwa.get("apmtidx", 2 )

    csgpath = kwa.get("csgpath", "$TMP/tboolean-csg-pmt-py")
    #gltfpath = kwa.get("gltfpath", "$TMP/tgltf/tgltf-gdml--.gltf")
    container = kwa.get("container","Rock//perfectAbsorbSurface/Vacuum") 
    testobject = kwa.get("testobject","Vacuum///GlassSchottF2" ) 
    gsel = kwa.get("gsel", "/dd/Geometry/PMT/lvPmtHemi0x" ) 
    gidx = kwa.get("gidx", 0 ) 
    gmaxnode = kwa.get("gmaxnode", 0 ) 
    gmaxdepth = kwa.get("gmaxdepth", 0 ) 
    cfordering = kwa.get("cfordering", "self" ) 


    parser = argparse.ArgumentParser(doc)

    parser.add_argument(     "--tag",  default=tag, help="tag identifiying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent. Default %(default)s" )
    parser.add_argument(     "--det",  default=det, help="detector geometry: eg PmtInBox, dayabay. Default %(default)s. "  )
    parser.add_argument(     "--src",  default=src, help="photon source: torch, scintillation OR cerenkov. Default %(default)s " )

    parser.add_argument(     "--noshow",  dest="show", default=show, action="store_false", help="switch off dumping commandline "  )
    parser.add_argument(     "--noplot",  dest="plot", default=plot, action="store_false", help="switch off plotting"  )
    parser.add_argument(     "--show",  default=show, action="store_true", help="dump invoking commandline "  )
    parser.add_argument(     "--loglevel", default=llv, help=" set logging level : DEBUG/INFO/WARNING/ERROR/CRITICAL. Default %(default)s." )

    parser.add_argument(     "--profile",  default=None, help="Unused option allowing argparser to cope with remnant ipython profile option"  )
    parser.add_argument(     "-i", dest="interactive", action="store_true", default=False, help="Unused option allowing argparser to cope with remnant ipython -i option"  )

    parser.add_argument(     "--tagoffset",  default=tagoffset, type=int, help="tagoffset : unsigned offset from tag, identifies event in multivent running. Default %(default)s "  )
    parser.add_argument(     "--multievent",  default=multievent, type=int, help="multievent : unsigned number of events to handle. Default %(default)s "  )
    parser.add_argument(     "--stag",  default=stag, help="S-Polarization tag : identifying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent" )
    parser.add_argument(     "--ptag",  default=ptag, help="P-Polarization tag : identifying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent" )
    parser.add_argument(     "--mrc",  default=mrc, type=int, help="script return code resulting from missing event files. Default %(default)s "  )
    parser.add_argument(     "--mat",  default=mat, help="material name, used for optical property dumping/plotting. Default %(default)s"  )
    parser.add_argument(     "--sli",  default=sli, help="slice specification delimited by colon. Default %(default)s"  )
    parser.add_argument(     "--sel",  default=sel, help="selection slice specification delimited by colon. Default %(default)s"  )
    parser.add_argument(     "--qwn",  default=qwn, help="Quantity by single char, pages delimited by comma eg XYZT,ABCR. Default %(default)s"  )
    parser.add_argument(     "--c2max",  default=c2max, type=float, help="Admissable total chi2 deviation in comparisons. Default %(default)s"  )
    parser.add_argument(     "--pfxseqhis",  default=pfxseqhis, help="Seqhis hexstring prefix for spawned selection. Default %(default)s"  )
    parser.add_argument(     "--pfxseqmat",  default=pfxseqmat, help="Seqmat hexstring prefix for spawned selection. Default %(default)s"  )
    parser.add_argument(     "--dbgseqhis",  default=dbgseqhis, help="Seqhis hexstring prefix for dumping. Default %(default)s"  )
    parser.add_argument(     "--dbgseqmat",  default=dbgseqmat, help="Seqmat hexstring prefix for dumping. Default %(default)s"  )
    parser.add_argument(     "--dbgmskhis",  default=dbgmskhis, help="History mask hexstring for selection/dumping. Default %(default)s"  )
    parser.add_argument(     "--dbgmskmat",  default=dbgmskmat, help="Material mask hexstring for selection/dumping. Default %(default)s"  )
    parser.add_argument(     "--figsize",  default=figsize, help="Comma delimited figure width,height in inches. Default %(default)s"  )
    parser.add_argument(     "--size",  default=size, help="Comma delimited figure width,height in inches. Default %(default)s"  )
    parser.add_argument(     "--dbgzero",  default=dbgzero, action="store_true", help="Dump sequence lines with zero counts. Default %(default)s"  )
    parser.add_argument(     "--terse", action="store_true", help="less verbose, useful together with --multievent ")
    parser.add_argument(     "--nosmry", dest="smry", action="store_false", help="nosmry option gives more detailed seqmat and seqhis tables, including the hex strings, useful for dbgseqhis")
    parser.add_argument(     "--pybnd",  action="store_true", help="Avoid error from op binary selection flag. ")
    parser.add_argument(     "--gdml2gltf",  action="store_true", help="Avoid error from op binary selection flag. ")
    parser.add_argument(     "--prohis", default=prohis, action="store_true", help="Present progressively masked seqhis frequency tables for step by step checking. Default %(default)s ")
    parser.add_argument(     "--promat", default=promat, action="store_true", help="Present progressively masked seqmat frequency tables for step by step checking. Default %(default)s ")
    parser.add_argument(     "--rehist", default=rehist, action="store_true", help="Recreate hists rather than loading persisted ones. Default %(default)s ")
    parser.add_argument(     "--chi2sel", default=chi2sel, action="store_true", help="Select histograms by their chi2 sum exceeding a cut, see cfh.py. Default %(default)s ")
    parser.add_argument(     "--chi2selcut", default=chi2selcut, type=float, help="chi2 per degree of freedom cut used to select histograms when using --chi2sel option, see cfh-vi. Default %(default)s ")
    parser.add_argument(     "--statcut", default=statcut, type=int, help="Statistics cut used with --chi2sel option, see cfh-vi Default %(default)s ")
    parser.add_argument(     "--nointerpol", dest="interpol", default=not nointerpol, action="store_false", help="See cfg4/tests/CInterpolationTest.py. Default %(default)s ")
    parser.add_argument(     "--lmx",  default=lmx, type=int, help="Maximum number of lines to present in sequence frequency tables. Default %(default)s "  )
    parser.add_argument(     "--cmx",  default=cmx, type=float, help="When greater than zero used as minimum line chi2 to present in sequence frequency tables. Default %(default)s "  )
    parser.add_argument(     "--apmtpathtmpl", default=apmtpathtmpl, help="Template Path to analytic PMT serialization, see pmt- and ana/pmt/analytic.py. %(default)s ")
    parser.add_argument(     "--apmtidx",      default=apmtidx, type=int, help="PmtPath index used to fill in the template, see pmt- and ana/pmt/analytic.py. %(default)s ")
    parser.add_argument(     "--apmtddpath",   default=apmtddpath, help="Path to detdesc xml file with description of DayaBay PMT, which references other files. %(default)s ")
    parser.add_argument(     "--addpath",   default=addpath, help="Path to detdesc xml file for topdown testing. %(default)s ")
    parser.add_argument(     "--yes", action="store_true", help="Confirm any YES dialogs. %(default)s ")
    parser.add_argument(     "--csgpath",   default=csgpath, help="Directory of the NCSG input serialization. %(default)s ")
    #parser.add_argument(     "--gltfpath",   default=gltfpath, help="Path to glTF json file. %(default)s ")
    parser.add_argument(     "--container",   default=container, help="Boundary specification for container. %(default)s ")
    parser.add_argument(     "--testobject",  default=testobject, help="Boundary specification for testobject. %(default)s ")
    parser.add_argument(     "--cfordering",  default=cfordering, help="Sort ordering of cf tables, one of max/self/other. %(default)s ")

    parser.add_argument(     "--gsel",  default=gsel, help="GDML node selection, either tree node index integer or LV name prefix, see tboolean-gdml . %(default)s ")
    parser.add_argument(     "--gmaxdepth",  default=gmaxdepth, type=int, help="GDML node depth limit, 0 for no limit, see tboolean-gdml. %(default)s ")
    parser.add_argument(     "--gmaxnode",  default=gmaxnode, type=int, help="GDML node limit including target node, 0 for no limit, see tboolean-gdml. %(default)s ")
    parser.add_argument(     "--gidx",  default=gidx, type=int, help="GDML index to pick target node from within gsel lvn selection, see tboolean-gdml. %(default)s ")
    parser.add_argument(     "--gltfsave", default=gltfsave, action="store_true", help="Save GDML parsed scene as glTF, see analytic/sc.py. %(default)s ")
    parser.add_argument(     "--lvnlist", default=lvnlist, help="Path to file containing list of lv names. %(default)s ")
    parser.add_argument(     "--j1707", action="store_true", help="Bash level option passthru. %(default)s ")
    parser.add_argument(     "--extras", action="store_true", help="Bash level option passthru. %(default)s ")
    parser.add_argument(     "--disco", action="store_true", help="Disable container, investigate suspected  inefficient raytrace of objects inside spacious containers. %(default)s ")

    parser.add_argument('nargs', nargs='*', help='nargs : non-option args')


    ok = OK()
    args = parser.parse_args(namespace=ok)
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()), format=fmt)

    if args.multievent > 1 and args.tagoffset > 0:
         log.fatal("use either --multievent n or --tagoffset o to pick one from multi, USING BOTH --multievent and --tagoffset NOT SUPPORTED  ") 
         sys.exit(1)

    assert args.cfordering in "max self other".split() 


    if args.multievent > 1:
        args.utags =  map(lambda offset:int(args.tag) + offset, range(args.multievent)) 
        args.utag = args.utags[0]   # backward compat for scripts not supporting multievent yet 
    else:
        try:
           tag = int(args.tag)
        except ValueError:
           tag = map(int,args.tag.split(","))
        pass

        if type(tag) is int:
            args.utag = tag + args.tagoffset 
            args.utags = [args.utag]   
        else:
            args.utag = None
            args.utags = tag  
        pass
    pass

    args.qwns = args.qwn.replace(",","")       
    os.environ.update(OPTICKS_MAIN_QWNS=args.qwns)  # dirty trick to avoid passing ok to objects that need this

    # hexstring -> hexint 
    args.dbgseqhis = int(str(args.dbgseqhis),16) 
    args.dbgseqmat = int(str(args.dbgseqmat),16) 
    args.dbgmskhis = int(str(args.dbgmskhis),16) 
    args.dbgmskmat = int(str(args.dbgmskmat),16) 
    args.figsize = map(float, args.figsize.split(","))

    args.sli = slice(*map(lambda _:int(_) if len(_) > 0 else None,args.sli.split(":")))
    args.sel = slice(*map(lambda _:int(_) if len(_) > 0 else None,args.sel.split(":")))



    args.apmtpath = args.apmtpathtmpl % dict(apmtidx=str(args.apmtidx))
    log.debug("args.apmtpathtmpl %s args.apmtidx %d -> args.apmtpath %s " % ( args.apmtpathtmpl, args.apmtidx, args.apmtpath ) ) 


    log.debug("args.dbgseqhis [%x] " % args.dbgseqhis) 

    if args.show:
         sys.stderr.write("args: " + " ".join(sys.argv) + "\n")

    #return args 

    ok.query = OpticksQuery(os.environ.get("OPTICKS_QUERY",""))

    return ok 

    
 


def ihex_(i):
    """
    # trim the 0x and L from a hex string
    """
    xs = hex(i)[2:]
    xs = xs[:-1] if xs[-1] == 'L' else xs 
    return xs 

_ini = {}
def ini_(path):
    global _ini 
    if _ini.get(path,None):
        log.debug("return cached ini for key %s" % path)
        return _ini[path] 
    try: 
        log.debug("parsing ini for key %s" % path)
        txt = file(os.path.expandvars(os.path.expanduser(path))).read()
        lines = filter(None, txt.split("\n"))
        d = dict(map(lambda _:_.split("="), lines))
        _ini[path] = d
    except IOError:
        log.fatal("failed to load ini from %s" % path)
        _ini[path] = {}
    pass
    return _ini[path] 


_json = {}
def json_(path):
    global _json 
    if _json.get(path,None):
        log.debug("return cached json for key %s" % path)
        return _json[path] 
    try: 
        log.debug("parsing json for key %s" % path)
        _json[path] = json.load(file(os.path.expandvars(os.path.expanduser(path))))
    except IOError:
        log.warning("failed to load json from %s" % path)
        _json[path] = {}
    pass
    return _json[path] 


_enum = {}
def enum_(path):
    global _enum
    if _enum.get(path, None):
        log.debug("return cached enum for key %s" % path)
        return _enum[path] 
    try:
        log.debug("parsing enum for key %s" % path)
        d = Enum(path)
        _enum[path] = d 
    except IOError:
        log.fatal("failed to load enum from %s" % path)
        _enum[path] = {}
    pass
    return _enum[path] 


class Abbrev(object):
    def __init__(self, path="$OPTICKS_DATA_DIR/resource/GFlags/abbrev.json"):
        js = json_(path)

        names = map(str,js.keys())
        abbrs = map(str,js.values())

        self.names = names
        self.abbrs = abbrs
        self.name2abbr = dict(zip(names, abbrs))
        self.abbr2name = dict(zip(abbrs, names))


class ItemList(object): # formerly ListFlags
    @classmethod 
    def Path(cls, txt, reldir=None):
        """
        :param txt: eg GMaterialLib
        :param reldir:  normally relative to IDPATH, for test geometry provide an absolute path
        """
        if reldir is not None and reldir.startswith("/"):
            npath = os.path.join(reldir, txt+".txt" )
        else:
            if reldir is None: 
                reldir = "GItemList"  
            pass 
            npath=idp_("%(reldir)s/%(txt)s.txt" % locals())
        pass
        return npath

    def __init__(self, txt="GMaterialLib", offset=1, translate_=None, reldir=None):
        """
        :param reldir: when starts with "/" an absolute path is assumed
        """
        npath=self.Path(txt, reldir)
        names = map(lambda _:_[:-1],file(npath).readlines())
        if translate_ is not None:
            log.info("translating")
            names = map(translate_, names) 
        pass
        codes = map(lambda _:_ + offset, range(len(names)))

        self.npath = npath
        self.offset = offset 
        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes)) 
        self.code2name = dict(zip(codes, names))

    def find_index(self, name):
        return self.names.index(name)

    def __str__(self):
        return "ItemLists names %6d name2code %6d code2name %6d offset %5d npath %s " % (len(self.names), len(self.name2code), len(self.code2name), self.offset, uidp_(self.npath))

    __repr__ = __str__



class IniFlags(object):
    """
    """
    def __init__(self, path="$OPTICKS_INSTALL_CACHE/OKC/GFlagIndexLocal.ini"):
        ini = ini_(path)
        assert len(ini) > 0, "IniFlags bad path/flags %s " % path 

        ini = dict(zip(ini.keys(),map(int,ini.values())))  # convert values to int 
        names = map(str,ini.keys())
        codes = map(int,ini.values())

        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes)) 
        self.code2name = dict(zip(codes, names))

class EnumFlags(object):
    def __init__(self, path="$OPTICKS_HOME/optickscore/OpticksPhoton.h"): 
        d = enum_(path) 
        ini = dict(zip(d.keys(),map(int,d.values())))  

        names = map(str,ini.keys())
        codes = map(int,ini.values())

        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes)) 
        self.code2name = dict(zip(codes, names))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    opticks_environment(dump=True)

    lf = ItemList("GMaterialLib")
    print "ItemList(GMaterialLib).name2code"
    print "\n".join([" %s : %s " % (k,v) for k,v in lf.name2code.items()])

    inif = IniFlags()
    print "IniFlags(photon flags)"
    print "\n".join([" %s : %s " % (k,v) for k,v in inif.name2code.items()])

    enuf = EnumFlags()
    print "EnumFlags(photon flags)"
    print "\n".join([" %s : %s " % (k,v) for k,v in enuf.name2code.items()])


