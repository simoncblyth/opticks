#!/usr/bin/env python
"""
"""

import numpy as np
import os, logging
from opticks.ana.bpath import BPath
from opticks.ana.key import Key

log = logging.getLogger(__name__) 


def _dirname(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def opticks_environment(dump=False):
   log.debug(" ( opticks_environment") 
   env = OpticksEnv()
   if dump:
       env.dump()
   env.bash_export() 
   log.debug(" ) opticks_environment") 


class OpticksEnv(object):
    """
    TODO: dependency on IDPATH when loading evt is dodgy as its making 
          the assumption that the geocache that IDPATH points at matches
          the geocache of the loaded evt...  

          this is true for test geometries too, as they still have a basis geocache

    FIX: by recording the geocache dir with the evt (is probably already is ?) 
         and be match asserted upon on load, dont just rely upon it 

    """
    def _idfilename(self):
        """
        layout 0
            /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae
            -> g4_00.dae

        layout > 0
             /usr/local/opticks/geocache/juno1707/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1
            -> g4_00.dae
     
        """
        log.debug("_opticks_idfilename layout %d : idpath %s " % (self.layout, self.idpath ))
        if self.layout == 0:
            name = os.path.basename(self.idpath)
            elem = name.split(".")
            assert len(elem) == 3 , ("form of idpath inconsistent with layout ", self.layout, self.idpath)
            idfilename = "%s.%s" % (elem[0],elem[2])
        elif self.layout > 0:
            idfilename = os.path.basename(os.path.dirname(os.path.dirname(self.idpath)))
        else:
            assert False, (self.layout, "unexpected layout") 
        pass
        log.debug("_opticks_idfilename layout %d : idpath %s -> idfilename %s " % (self.layout, self.idpath, idfilename))
        return idfilename

    def _idname(self):
        """
        idpath0="/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae"
        idpath1="/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1"

        -> 'DayaBay_VGDX_20140414-1300'
        """ 
        if self.layout == 0:
            idname  = self.idpath.split("/")[-2]
        else:
            idname  = self.idpath.split("/")[-4]
        pass
        return idname

    def _detector(self):
        """
        does not make any sense in direct approach
        """
        idname = self._idname()
        dbeg = idname.split("_")[0]
        if dbeg in ["DayaBay","LingAo","Far"]:
            detector =  "DayaBay"
        else:
            detector = dbeg 
        pass
        log.debug("_opticks_detector idpath %s -> detector %s " % (self.idpath, detector))
        return detector 

    def _detector_dir(self):
        """
        in layout 1, this yields /usr/local/opticks/opticksdata/export/juno1707/
        but should be looking in IDPATH ?
        """
        detector = self._detector()
        return os.path.join(self.env["OPTICKS_EXPORT_DIR"], detector)


    def __init__(self, legacy=False):
        self.ext = {}
        self.env = {}

        if os.environ.has_key("IDPATH"):
            self.legacy_init()
        else:
            self.direct_init()
        pass


    def direct_init(self): 
        """
        Direct approach

        * IDPATH is not allowed as an input, it is an internal envvar only 

        """ 
        assert not os.environ.has_key("IDPATH"), "IDPATH envvar as input is forbidden"
        assert os.environ.has_key("OPTICKS_KEY"), "OPTICKS_KEY envvar is required"
        self.key = Key(os.environ["OPTICKS_KEY"])

        keydir = self.key.keydir 
        assert os.path.isdir(keydir), "keydir %s is required to exist " % keydir  

        ## not defaults 
        os.environ["IDPATH"] = keydir   ## <-- to be removed, switch to GEOCACHE signally direct workflow 
        os.environ["GEOCACHE"] = keydir    

        self.install_prefix = _dirname(keydir, 5)

        self.setdefault("OPTICKS_INSTALL_PREFIX",  self.install_prefix)
        self.setdefault("OPTICKS_INSTALL_CACHE",   os.path.join(self.install_prefix, "installcache"))

        #self.setdefault("OPTICKS_EVENT_BASE",      os.path.join(keydir, "source" ))
        self.setdefault("OPTICKS_EVENT_BASE",       keydir )


    def legacy_init(self): 
        assert os.environ.has_key("IDPATH"), "IDPATH envvar is required, for legacy running"
        if os.environ.has_key("OPTICKS_KEY"):
            del os.environ['OPTICKS_KEY']
            log.warning("legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init" )
        pass

        IDPATH = os.environ["IDPATH"] 
        os.environ["GEOCACHE"] = IDPATH   ## so can use in legacy approach too    

        self.idpath = IDPATH

        self.idp = BPath(IDPATH)
        self.srcpath = self.idp.srcpath
        self.layout  = self.idp.layout

        idfold = _dirname(IDPATH,1)
        idfilename = self._idfilename()

        self.setdefault("OPTICKS_IDPATH",          IDPATH )
        self.setdefault("OPTICKS_IDFOLD",          idfold )
        self.setdefault("OPTICKS_IDFILENAME",      idfilename )

        if self.layout == 0:  
            self.install_prefix = _dirname(IDPATH, 4)
            self.setdefault("OPTICKS_DAEPATH",         os.path.join(idfold, idfilename))
            self.setdefault("OPTICKS_GDMLPATH",        os.path.join(idfold, idfilename.replace(".dae",".gdml")))
            self.setdefault("OPTICKS_GLTFPATH",        os.path.join(idfold, idfilename.replace(".dae",".gltf")))
            self.setdefault("OPTICKS_DATA_DIR",        _dirname(IDPATH,3))
            self.setdefault("OPTICKS_EXPORT_DIR",      _dirname(IDPATH,2))
        else:
            self.install_prefix = _dirname(IDPATH, 5)
            self.setdefault("OPTICKS_DAEPATH",         self.srcpath)
            self.setdefault("OPTICKS_GDMLPATH",        self.srcpath.replace(".dae",".gdml"))
            self.setdefault("OPTICKS_GLTFPATH",        self.srcpath.replace(".dae",".gltf"))
            self.setdefault("OPTICKS_DATA_DIR",        _dirname(self.srcpath,3))     ## HUH thats a top down dir, why go from bottom up for it ?
            self.setdefault("OPTICKS_EXPORT_DIR",      _dirname(self.srcpath,2))
        pass

        self.setdefault("OPTICKS_INSTALL_PREFIX",  self.install_prefix)
        self.setdefault("OPTICKS_INSTALL_CACHE",   os.path.join(self.install_prefix, "installcache"))
        log.debug("install_prefix : %s " % self.install_prefix ) 

        self.setdefault("OPTICKS_DETECTOR",        self._detector())
        self.setdefault("OPTICKS_DETECTOR_DIR",    self._detector_dir())
        self.setdefault("OPTICKS_EVENT_BASE",      os.path.expandvars("/tmp/$USER/opticks") )
        self.setdefault("OPTICKS_QUERY",           "")
        self.setdefault("TMP",                     os.path.expandvars("/tmp/$USER/opticks") )

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
        """
        If the key is already in the environment does not 
        change it, just records into ext 
        """
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



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment(dump=True)



