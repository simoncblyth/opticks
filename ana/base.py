#!/usr/bin/env python
"""
Mostly Non-numpy basics, just numpy configuration


"""

import numpy as np
import os, logging, json, ctypes, subprocess, datetime, re
from collections import OrderedDict as odict 
from opticks.ana.enum import Enum 

log = logging.getLogger(__name__) 


try:
    cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
    ffs_ = lambda _:cpp.ffs(_)
except OSError:
    pass



idp_ = lambda _:"%s/%s" % (os.environ["IDPATH"],_) 
uidp_ = lambda _:_.replace(os.environ["IDPATH"],"$IDPATH")

gcp_ = lambda _:"%s/%s" % (os.environ["GEOCACHE"],_) 




def findfile(base, name, relative=True):
    paths = []
    for root, dirs, files in os.walk(base):
        if name in files: 
            path = os.path.join(root,name)
            paths.append(path[len(base)+1:] if relative else path)
        pass
    pass 
    return paths


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
    print(fmt % dict(zip(keys,keys)))

    for idx in idxs:
        meta = json_load_(os.path.join(base,str(idx),name))
        meta['idx'] = idx
        meta['err'] = meta.get('err',"-")
        print(fmt % meta)
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

def _opticks_env(st="OPTICKS_ IDPATH"):
    return filter(lambda _:_[0].startswith(st.split()), os.environ.items())


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
        assert 0, ( path )
        _ini[path] = {}
    pass
    return _ini[path] 


_json = {}
def json_(path):
    global _json 
    if _json.get(path,None):
        log.debug("return cached json for key %s" % path)
        return _json[path]
        return js
    try: 
        log.debug("parsing json for key %s" % path)
        xpath = os.path.expandvars(os.path.expanduser(path))
        #log.info("xpath:%s"%xpath)
        js = json.load(file(xpath))
        js[u"jsonLoadPath"] = unicode(xpath) 
        _json[path] = js 
    except IOError:
        log.warning("failed to load json from %s : %s " % (path,xpath))
        assert 0
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
    """
    simon:opticksdata blyth$ find . -name abbrev.json
    ./export/DayaBay/GMaterialLib/abbrev.json
    ./resource/GFlags/abbrev.json
    simon:opticksdata blyth$ 

    $OPTICKS_DATA_DIR/resource/GFlags/abbrev.json
    """
    def __init__(self, path):
        js = json_(path)
    
        if "abbrev" in js:
            js = js["abbrev"]
        pass
        #log.info("js:%s" % js.keys() )

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
        log.debug("txt %s reldir  %s " % (txt, reldir))
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

        #assert 0, "who uses this ?" 

        ini = dict(zip(ini.keys(),map(int,ini.values())))  # convert values to int 
        names = map(str,ini.keys())
        codes = map(int,ini.values())

        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes)) 
        self.code2name = dict(zip(codes, names))

class EnumFlags(object):
    """
    With the default of mask2int False the values are::
    
        1 << 0, 1 << 1, 1 << 2, ...

    Otherwise with mask2int True they are::

        1,2,3,...
 
    """
    def __init__(self, path, mask2int=False): 
        d = enum_(path) 
        ini = dict(zip(d.keys(),map(int,d.values())))  

        names = map(str,ini.keys())
        codes = map(int,ini.values())

        if mask2int:
            mask2int = {}
            for i in range(32):
                mask2int[1 << i] = i + 1 
            pass 
            codes = map(lambda _:mask2int.get(_,-1), codes)
        pass

        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes)) 
        self.code2name = dict(zip(codes, names))

   
class PhotonMaskFlags(EnumFlags):
    """
    Note this is partially duplicating optickscore/OpticksFlags.cc 

    Abbrev used to come "$OPTICKS_DATA_DIR/resource/GFlags/abbrev.json"
    """
    def __init__(self):
        EnumFlags.__init__(self, path="$OPTICKS_HOME/optickscore/OpticksPhoton.h", mask2int=False) 
        self.abbrev = Abbrev("$OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json")


class PhotonCodeFlags(EnumFlags):
    """
    """
    def __init__(self):
        EnumFlags.__init__(self, path="$OPTICKS_HOME/optickscore/OpticksPhoton.h", mask2int=True) 
        self.abbrev = Abbrev("$OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json")




if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main() 


    lf = ItemList("GMaterialLib")
    print("ItemList(GMaterialLib).name2code")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(lf.name2code.items(),key=lambda kv:kv[1])]))

    inif = IniFlags()
    print("IniFlags(photon flags)")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(inif.name2code.items(),key=lambda kv:kv[1])]))

    phmf = PhotonMaskFlags()
    print("PhotonMaskFlags()")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(phmf.name2code.items(),key=lambda kv:kv[1])]))

    phcf = PhotonCodeFlags()
    print("PhotonCodeFlags()")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(phcf.name2code.items(),key=lambda kv:kv[1])]))



