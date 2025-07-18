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
Strictly Non-numpy basics


"""

import os, logging, json, ctypes, subprocess, datetime, re
log = logging.getLogger(__name__)

from collections import OrderedDict as odict
import numpy as np
from opticks.ana.enum_ import Enum

#from opticks.ana.key import keydir
#KEYDIR = keydir()

log = logging.getLogger(__name__)


try:
    cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
    ffs_ = lambda _:cpp.ffs(_)
except OSError:
    pass


#idp_ = lambda _:"%s/%s" % (os.environ["IDPATH"],_)
#uidp_ = lambda _:_.replace(os.environ["IDPATH"],"$IDPATH")
#
#idp_ = lambda _:"%s/%s" % (KEYDIR,_)
#uidp_ = lambda _:_.replace(KEYDIR,"$KEYDIR")

gcp_ = lambda _:"%s/%s" % (os.environ["GEOCACHE"],_)


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


splitlines_ = lambda txtpath:open(txtpath).read().splitlines()

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

    keys = re.compile(r"\((\w*)\)").findall(fmt)
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
json_load_ = lambda path:json.load(open(expand_(path)))
json_save_ = lambda path, d:json.dump(d, open(makedirs_(expand_(path)),"w"), cls=NPEncoder)
json_save_pretty_ = lambda path, d:json.dump(d, open(makedirs_(expand_(path)),"w"),cls=NPEncoder, sort_keys=True, indent=4, separators=(',', ': '))

class NPEncoder(json.JSONEncoder):
    """
    Custom encoder for numpy data types
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)




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
    Trim the 0x and any L from a hex string::

        assert ihex_(0xccd) == 'ccd'
        assert ihex_(0xccdL) == 'ccd'    # assumed

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
        xpath = os.path.expandvars(os.path.expanduser(path))
        txt = open(xpath,"r").read()
        lines = list(filter(None, txt.split("\n")))
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
        js = json.load(open(xpath,"r"))
        #js[u"jsonLoadPath"] = unicode(xpath)
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

        names = list(map(str,js.keys()))    # CAUTION ORDER DOES NOT MATCH PhotonCodeFlags
        abbrs = list(map(str,js.values()))

        self.names = names
        self.abbrs = abbrs
        self.name2abbr = dict(zip(names, abbrs))
        self.abbr2name = dict(zip(abbrs, names))


    def __repr__(self):
        lines = []
        lines.append(".names")
        lines.extend(self.names)
        lines.append(".abbrs")
        lines.extend(self.abbrs)
        return "\n".join(lines)


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
        names = list(map(lambda _:_[:-1],open(npath,"r").readlines()))
        if translate_ is not None:
            log.info("translating")
            names = list(map(translate_, names))
        pass
        codes = list(map(lambda _:_ + offset, range(len(names))))

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
    Formerly from $OPTICKS_INSTALL_CACHE/OKC/GFlagIndexLocal.ini
    """
    def __init__(self, path="$OPTICKS_PREFIX/include/SysRap/OpticksPhoton_Enum.ini"):
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
        ini = dict(zip(d.keys(),list(map(int,d.values()))))

        names = list(map(str,ini.keys()))
        codes = list(map(int,ini.values()))

        if mask2int:
            mask2int = {}
            for i in range(32):
                mask2int[1 << i] = i + 1
            pass
            codes = list(map(lambda _:mask2int.get(_,-1), codes))
        pass

        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes))
        self.code2name = dict(zip(codes, names))

    def __repr__(self):
        return "\n".join([self.__class__.__name__, "names", str(self.names), "codes", str(self.codes), "name2code", str(self.name2code), "code2name", str(self.code2name) ])

class PhotonMaskFlags(EnumFlags):
    """
    This is used by hismask.py for pflags_ana

    Note this is partially duplicating optickscore/OpticksFlags.cc
    Former positions of Abbrev : $OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json
    """
    def __init__(self):
        EnumFlags.__init__(self, path="$OPTICKS_PREFIX/include/SysRap/OpticksPhoton.h", mask2int=False)
        self.abbrev = Abbrev("$OPTICKS_PREFIX/include/SysRap/OpticksPhoton_Abbrev.json")
        ## looks like SysRap suffers case inconsistency



class PhotonCodeFlags(EnumFlags):
    """
    This is used by histype.py for seqhis_ana
    """
    def __init__(self):
        EnumFlags.__init__(self, path="$OPTICKS_PREFIX/include/SysRap/OpticksPhoton.h", mask2int=True)

        abbrev = Abbrev("$OPTICKS_PREFIX/include/SysRap/OpticksPhoton_Abbrev.json")
        name2abbr = abbrev.name2abbr
        names = self.names

        abbr2code = {}
        abbr = []
        for code, name in enumerate(names):
            abr = name2abbr.get(name, "??")
            abbr.append(abr)
            abbr2code[abr] = code + 1
        pass

        fln = np.array(["~~"]+names)
        fla = np.array(["~~"]+abbr)
        ftab = np.c_[np.arange(len(fla)),fla,fln]

        self.abbrev = abbrev
        self.abbr = abbr      # these abbr follow same order as names unlike abbrev.names
        self.fln = fln
        self.fla = fla
        self.ftab = ftab
        self.abbr2code = abbr2code

    def __getattr__(self, arg):
        code = self.abbr2code.get(arg, -1)
        return code



def test_basics():
    from opticks.ana.main import opticks_main
    ok = opticks_main()

    lf = ItemList("GMaterialLib")
    print("lf:ItemList(GMaterialLib).name2code")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(lf.name2code.items(),key=lambda kv:kv[1])]))

    inif = IniFlags()
    print("inif:IniFlags(photon flags)")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(inif.name2code.items(),key=lambda kv:kv[1])]))

    phmf = PhotonMaskFlags()
    print("phmf:PhotonMaskFlags()")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(phmf.name2code.items(),key=lambda kv:kv[1])]))

    phcf = PhotonCodeFlags()
    print("phcf:PhotonCodeFlags()")
    print("\n".join([" %30s : %s " % (k,v) for k,v in sorted(phcf.name2code.items(),key=lambda kv:kv[1])]))



if __name__ == '__main__':

    pcf = PhotonCodeFlags()
    print(pcf.ftab)





