#!/usr/bin/env python
"""
Non-numpy basics
"""
import os, logging, json, ctypes
log = logging.getLogger(__name__) 

cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
ffs_ = lambda _:cpp.ffs(_)

IDPATH = os.path.expandvars("$IDPATH")
idp_ = lambda _:"%s/%s" % (IDPATH,_) 

def _dirname(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path

def _opticks_install_prefix(idpath):
    prefix = _dirname(idpath,4)
    log.info("_opticks_install_prefix idpath %s -> prefix %s " % (idpath, prefix))
    return prefix 

def _opticks_data(idpath):
    datadir = _dirname(idpath,3)
    log.info("_opticks_datadir idpath %s -> datadir %s " % (idpath, datadir))
    return datadir 

def _opticks_install_cache(idpath):
    prefix = _opticks_install_prefix(idpath) 
    path = os.path.join(prefix, "installcache") 
    log.info("_opticks_install_cache idpath %s -> datadir %s " % (idpath, path))
    return path 




def _opticks_detector(idpath):
    ddir = idpath.split("/")[-2]
    dbeg = ddir.split("_")[0]
    if dbeg in ["DayaBay","LingAo","Far"]:
        detector =  "DayaBay"
    else:
        detector = dbeg 
    pass
    log.info("_opticks_detector idpath %s -> detector %s " % (idpath, detector))
    return detector 







def _opticks_dump():
    for k,v in os.environ.items():
        if k.startswith("OPTICKS_"):
            log.info(" %30s : %s " % (k,v))
    
os.environ.setdefault("OPTICKS_DATA",            _opticks_data(IDPATH))
os.environ.setdefault("OPTICKS_INSTALL_PREFIX",  _opticks_install_prefix(IDPATH))
os.environ.setdefault("OPTICKS_INSTALL_CACHE",   _opticks_install_cache(IDPATH))
os.environ.setdefault("OPTICKS_DETECTOR",        _opticks_detector(IDPATH))


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
    def __init__(self, path="$OPTICKS_DATA/resource/GFlags/abbrev.json"):
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

    _opticks_dump()


    lf = ItemList("GMaterialLib")
    print "ItemList(GMaterialLib).name2code", lf.name2code

    inif = IniFlags()
    print "IniFlags(photon flags)", inif.name2code



