#!/usr/bin/env python
"""
Non-numpy basics
"""

import os, logging, json, ctypes, subprocess, argparse, sys
log = logging.getLogger(__name__) 


try:
    cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
    ffs_ = lambda _:cpp.ffs(_)
except OSError:
    pass

IDPATH = os.path.expandvars("$IDPATH")
idp_ = lambda _:"%s/%s" % (IDPATH,_) 

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

class OpticksEnv(object):
    def __init__(self):
        self.ext = {}
        self.env = {}

        if not os.path.isdir(IDPATH): 
            print "Invalid/missing IDPATH envvar %s " % IDPATH
            sys.exit(1)  
 
        self.setdefault("OPTICKS_IDFOLD",          _opticks_idfold(IDPATH))
        self.setdefault("OPTICKS_IDFILENAME",      _opticks_idfilename(IDPATH))
        self.setdefault("OPTICKS_DAEPATH",         _opticks_daepath(IDPATH))
        self.setdefault("OPTICKS_DATA_DIR",        _opticks_data_dir(IDPATH))
        self.setdefault("OPTICKS_EXPORT_DIR",      _opticks_export_dir(IDPATH))
        self.setdefault("OPTICKS_INSTALL_PREFIX",  _opticks_install_prefix(IDPATH))
        self.setdefault("OPTICKS_INSTALL_CACHE",   _opticks_install_cache(IDPATH))
        self.setdefault("OPTICKS_DETECTOR",        _opticks_detector(IDPATH))
        self.setdefault("OPTICKS_DETECTOR_DIR",    _opticks_detector_dir(IDPATH))
        self.setdefault("OPTICKS_EVENT_BASE",      _opticks_event_base())
        self.setdefault("TMP",                     _opticks_tmp())

    def bash_export(self, path="$TMP/opticks_env.bash"):
        lines = []
        for k,v in self.env.items():
            line = "export %s=%s " % (k,v)
            lines.append(line)    

        path = os.path.expandvars(path) 
        print "writing opticks environment to %s " % path 
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

def opticks_args(**kwa):

    llv = kwa.get("loglevel", "info")
    doc = kwa.get("doc", None)
    tag = kwa.get("tag", None)
    stag = kwa.get("stag", None)
    ptag = kwa.get("ptag", None)
    src = kwa.get("src", None)
    det = kwa.get("det", None)
    typ = kwa.get("typ", None)
    show = kwa.get("show", True)

    parser = argparse.ArgumentParser(doc)

    parser.add_argument(     "--noshow",  dest="show", default=show, action="store_false", help="switch off dumping commandline "  )
    parser.add_argument(     "--show",  default=show, action="store_true", help="dump invoking commandline "  )
    parser.add_argument(     "--loglevel", default=llv, help=" set logging level : DEBUG/INFO/WARNING/ERROR/CRITICAL " )

    parser.add_argument(     "--tag",  default=tag, help="tag identifiying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent" )
    parser.add_argument(     "--stag",  default=stag, help="S-Polarization tag : identifying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent" )
    parser.add_argument(     "--ptag",  default=ptag, help="P-Polarization tag : identifying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent" )
    parser.add_argument(     "--src",  default=src, help="photon source: torch, scintillation OR cerenkov " )
    parser.add_argument(     "--det",  default=det, help="detector geometry: eg PmtInBox, dayabay "  )
    parser.add_argument(     "--typ",  default=typ, help="photon source: eg torch, cerenkov, scintillation "  )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))

    if args.show:
         print " ".join(sys.argv)

    return args 

    
 


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
   def __init__(self, txt="GMaterialLib", offset=1 ):
        npath=idp_("GItemList/%(txt)s.txt" % locals())
        names = map(lambda _:_[:-1],file(npath).readlines())
        codes = map(lambda _:_ + offset, range(len(names)))
        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes)) 
        self.code2name = dict(zip(codes, names))

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




if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    opticks_environment(dump=True)

    lf = ItemList("GMaterialLib")
    print "ItemList(GMaterialLib).name2code"
    print "\n".join([" %s : %s " % (k,v) for k,v in lf.name2code.items()])

    inif = IniFlags()
    print "IniFlags(photon flags)"
    print "\n".join([" %s : %s " % (k,v) for k,v in inif.name2code.items()])



