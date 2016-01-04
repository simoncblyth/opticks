#!/usr/bin/env python
"""
Non-numpy basics
"""
import os, logging, json, ctypes
log = logging.getLogger(__name__) 

cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
ffs_ = lambda _:cpp.ffs(_)


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




if __name__ == '__main__':
    i = ini_("$IDPATH/GFlagsLocal.ini")
    j = json_("~/.opticks/GFlags/abbrev.json")
    n = ffs_(0x1000)

