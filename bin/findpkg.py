#!/usr/bin/env python
"""
findpkg.py
============

Note that the functionality here is used by::

     find_package.py 
     pkg_config.py 

After code changes make sure to do a --nocache or -C run with::

     find_package.py -C

Otherwise old errors from the cache may linger.

Hmm this depends on cmake modules from the opticks installed prefix
whereas it should in principal be useable prior to installing Opticks



"""
import os, re, logging, argparse, sys, json, platform
import shutil, tempfile, commands, stat, glob, fnmatch

try:
    from commands import getstatusoutput 
except ImportError:
    from subprocess import getstatusoutput 
pass 

from collections import OrderedDict as odict

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


log = logging.getLogger(__name__)

class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp() so it's usable with "with" statement."""
    def __enter__(self):
        self.iwd = os.getcwd()
        path = tempfile.mkdtemp()

        os.chdir(path)
        self.path = path
        return path

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.iwd)
        shutil.rmtree(self.path)

class Tmpl(dict):
    def __str__(self):
        return self.__doc__ % self
 
class CMakeLists(Tmpl):
    """
cmake_minimum_required(VERSION 3.5)
set(name Use%(pkg)s)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()

set(%(pkg)s_VERBOSE ON) 

find_package(%(pkg)s %(opts)s)

include(EchoTarget)

if(%(pkg)s STREQUAL "Boost")
set(tgt "Boost::boost")
else()
set(tgt Opticks::%(pkg)s)
endif()


echo_target_std(${tgt})

set(_props
   INTERFACE_AUTOUIC_OPTIONS
   INTERFACE_COMPILE_DEFINITIONS
   INTERFACE_COMPILE_FEATURES
   INTERFACE_COMPILE_OPTIONS
   INTERFACE_INCLUDE_DIRECTORIES
   INTERFACE_LINK_DEPENDS
   INTERFACE_LINK_DIRECTORIES
   INTERFACE_LINK_LIBRARIES
   INTERFACE_LINK_OPTIONS
   INTERFACE_PRECOMPILE_HEADERS
   INTERFACE_POSITION_INDEPENDENT_CODE
   INTERFACE_SOURCES
   INTERFACE_SYSTEM_INCLUDE_DIRECTORIES

   INTERFACE_FIND_PACKAGE_NAME
   INTERFACE_PKG_CONFIG_NAME
   INTERFACE_IMPORTED_LOCATION 

   INTERFACE_INSTALL_CONFIGFILE_BCM
   INTERFACE_INSTALL_CONFIGDIR_BCM
   INTERFACE_INSTALL_LIBDIR_BCM
   INTERFACE_INSTALL_INCLUDEDIR_BCM
   INTERFACE_INSTALL_PREFIX_BCM
)



set(ill)
if(TARGET ${tgt}) 
   foreach(prop ${_props})
      get_property(val TARGET ${tgt} PROPERTY ${prop})
      message(STATUS "${prop}=${val}")
   endforeach()
   get_property(ill TARGET ${tgt} PROPERTY INTERFACE_LINK_LIBRARIES)
endif()
unset(_props)


# old style CMake usage just sets variables
message(STATUS "DIR=${%(pkg)s_DIR}")
message(STATUS "PREFIX=${%(pkg)s_PREFIX}")
message(STATUS "LIBDIR=${%(pkg)s_LIBDIR}")
message(STATUS "MODULE=${%(pkg)s_MODULE}")
message(STATUS "LIBRARY=${%(pkg)s_LIBRARY}")
message(STATUS "LIBRARIES=${%(pkg)s_LIBRARIES}")
message(STATUS "DEFINITIONS=${%(pkg)s_DEFINITIONS}")
message(STATUS "INCLUDE_DIR=${%(pkg)s_INCLUDE_DIR}")
message(STATUS "INCLUDE_DIRS=${%(pkg)s_INCLUDE_DIRS}")
message(STATUS "USE_FILE=${%(pkg)s_USE_FILE}")

# hmm to get libdirs need to traverse the INTERFACE_LINK_LIBRARIES
# and lookup their properties

foreach(subtgt ${ill})
  if(TARGET ${subtgt}) 
     get_property(loc TARGET ${subtgt} PROPERTY INTERFACE_IMPORTED_LOCATION)
     message(STATUS "subtgt:${subtgt} loc:${loc}")
  else()
     message(STATUS "subtgt:${subtgt} not-target")
  endif()  
endforeach()


"""
          
class Script(Tmpl):
    """#!/bin/bash 

pwd
ls -l

source $OPTICKS_PREFIX/bin/opticks-setup.sh 
echo OPTICKS_PREFIX $OPTICKS_PREFIX
echo CMAKE_PREFIX_PATH 
echo $CMAKE_PREFIX_PATH | tr ":" "\n"

cat -n CMakeLists.txt

mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
    -DCMAKE_MODULE_PATH=$OPTICKS_HOME/cmake/Modules \
    -DOPTICKS_PREFIX=$OPTICKS_PREFIX



"""



class DirectPkg(odict): 
    def __str__(self):
        return "\n".join(["%-40s = %s" % tuple(kv) for kv in self.items()])

    def __repr__(self):
        return repr_pkg(self)

    @classmethod
    def DefaultTmpDir(cls):
        k = "USERNAME" if platform.system() == "Windows" else "USER"
        username = os.environ[k]
        return os.path.join("/tmp", username, "opticks")

    @classmethod
    def Path(cls, name):
        default_tmpdir = cls.DefaultTmpDir()
        tmpdir = os.environ.get("OPTICKS_TMP", default_tmpdir )
        log.debug("tmpdir %s " % tmpdir)
        return os.path.join(tmpdir, "bin/findpkg/%s.json" % name )

    @classmethod
    def Exists(cls, name):
        path = cls.Path(name)
        return os.path.exists(path)

    @classmethod
    def Save(cls, d):
        path = cls.Path(d["name"])
        fold = os.path.dirname(path)
        if not os.path.isdir(fold):
            log.debug("creating fold %s " % fold)
            os.makedirs(fold)
        pass    
        outpath = path.replace(".json",".out")
        log.debug("Save out %s " % outpath )
        file(outpath, "w").write(d["out"])
        d["out"] = outpath
        log.debug("Save json %s " % path )
        json_save_pretty_(path, d)

    @classmethod
    def Load(cls, name):
        path = cls.Path(name)
        outpath = path.replace(".json",".out")
        log.debug("Load %s " % path )
        d = json_load_(path)
        d["out"] = file(outpath, "r").read()
        return d

    name = property(lambda self:self.get("name",""))

    def mget(self, props):
        for prop in props:
            val = self.get(prop,"")
            if val != "":
               return val
            pass
        return ""  

    def _get_libdir(self):
        """
        not so easy as a variety of properties need to be consulted
        """
        return getlibdir(self.mget(["INTERFACE_IMPORTED_LOCATION","LIBRARY","USE_FILE","DIR","LIBDIR"]))
 
    def _get_prefix(self):
        return self.mget(["PREFIX"])  # eg G4_PREFIX

    def _get_includedir(self):
        return self.mget(["INCLUDE_DIR", "INTERFACE_INCLUDE_DIRECTORIES"])

    prefix = property(_get_prefix)
    libdir = property(_get_libdir)
    includedir = property(_get_includedir)









def getlibdir(path):
    """
    :return libdir: parent directory of path ending with lib or lib64 or blank if none found 
    """
    libs = ["lib","lib64"]
    fold, name = os.path.split(path) 
    if name in libs:
        return path
    pass 

    elem = fold.split("/")
    jlib = -1
    for i in range(len(elem)):
        j = len(elem)-i-1
        if elem[j] in libs:
            jlib = j
            break
        pass   
    pass
    return "/".join(elem[:jlib+1]) if jlib > -1 else ""  




def repr_pkg(pkg):
    return "%-30s : %-50s : %-50s " % (pkg.name, pkg.includedir, pkg.libdir)


class Compare(object):
    def __init__(self, al, aa, bl, bb ):
        rc = self.compare(al,aa,bl,bb)

    def compare(self, al,aa,bl,bb):
        self.n_all = len(aa)
        self.n_dif = 0

        an = map(lambda _:_.name, aa) 
        bn = map(lambda _:_.name, bb) 

        ab = set(an) - set(bn)
        ba = set(bn) - set(an)
      
        print("%s-%s %s " % (al,bl,ab))
        print("%s-%s %s " % (bl,al,ba))

        if len(aa) != len(bb):
            log.fatal("Different num_pkgs %s:%d %s:%d " % (al,len(aa), bl,len(bb)))
            return 1
        pass

        if len(ab) != 0 or len(ba) != 0:
            log.fatal("Different pkg names found by cm and pc " )
            return 2
        pass

        bb = sorted(bb, key=lambda pkg:an.index(pkg.name)) # sort bb into the same order as aa

        n_dif = 0 
        for i in range(n_all):
            a = aa[i]
            b = bb[i]
            df = Pkg.Equal(a,b)
            if len(df) > 0:
                log.fatal(" a and b differ %s\n%s:%s\n%s:%s" % (repr(df),al,repr(a), bl,repr(b)))
                n_dif += 1 
            else:
                pass
                #log.info(" a and b are equal\n%s:%s\n%s:%s" % (al,repr(a), bl,repr(b)))
            pass
        pass
        self.n_dif = n_dif

    def __repr__(self):
        return " %d / %d differences " % (self.n_dif, self.n_all)
        




class Pkg(odict):

    VAR = re.compile("^(?P<var>\S*)=(?P<val>.*)$")

    @classmethod 
    def CreateFromPC(cls, path):

        log.info("CreateFromPC %s " % path )
        name = os.path.basename(path)
        assert name.endswith(".pc")
        name = name[:-3]

        d = {}
        d["pcfiledir"] = os.path.dirname(path)  # clhep uses this key in prefix
        d["path"] = path 
        d["name"] = name

        for line in file(path, "r").readlines():
            m = cls.VAR.match(line)
            if not m: continue
            var = m.groupdict()["var"]
            val = m.groupdict()["val"]
            d[var] = val 
            d[var+"_"] = val   # record original without interpolation
        pass

        # interpolate the variables 
        for k,v in d.items():
            if k.endswith("_"): continue
            d[k] = v.replace("}",")s").replace("{","(").replace("$","%") % d
        pass

        return cls(d)

    @classmethod
    def Recover(cls,d):
        """
        # pc from geant4 has only prefix  
        But this doesnt help need to fix the problem at source as opticks-config 
        aka pkg-config needs to get the message   

        BUT all it needs is --cflags and --libs not the variables

        """
        if "prefix" in d and not ("libdir" in d and "includedir" in d):
            libdir = "%s/lib64" % d["prefix"]  
            includedir = "%s/include" % d["prefix"]  
            if os.path.isdir(libdir):
                d["libdir"] = libdir
            pass   
            if os.path.isdir(includedir):
                d["includedir"] = includedir
            pass   
        pass




    def _get_libdir(self):
        return self.get("libdir","")
    def _get_includedir(self):
        return self.get("includedir","")
    def _get_prefix(self):
        return self.get("prefix","")

    prefix = property(_get_prefix)
    includedir = property(_get_includedir)
    libdir = property(_get_libdir)

    path = property(lambda self:self["path"])
    name = property(lambda self:self["name"])

    def __eq__(self, other): 
        """
        NB the path is excluded, as need pc and cmake Pkgs to be equable
        """
        #if not isinstance(other, Pkg):
        #    return NotImplemented
        return self.name == other.name and self.libdir == other.libdir and self.prefix == other.prefix and self.includedir == other.includedir 
   
    @classmethod
    def Equal(cls, a, b):
        atts = "name libdir includedir".split()
        df = []
        for att in atts:
            if getattr(a, att) != getattr(b, att):
                df.append(att)
            pass
        pass
        if len(df) > 0:
            log.debug(repr(df))
        pass
        return df 

 
    def __repr__(self):
        return repr_pkg(self)


class Find(object):
    @classmethod
    def parse_args(cls, doc, default_mode="cmake"):
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "names", nargs="*", help="logging level" ) 
        parser.add_argument( "--level", default="info", help="logging level" ) 
        parser.add_argument( "-C", "--nocache", dest="cache",  default=True, action="store_false"  )
        parser.add_argument( "-p", "--prefix",  default=False, action="store_true"  )
        parser.add_argument( "-l", "--libdir",  default=False, action="store_true" )
        parser.add_argument( "-i", "--includedir",  default=False, action="store_true" )
        parser.add_argument( "-n", "--name",  default=False, action="store_true" )
        parser.add_argument( "-d", "--dump",  default=False, action="store_true", help="Dump extra details for debugging" )
        parser.add_argument( "-s", "--dumpnames",  default=False, action="store_true", help="Dump names for debugging" )
        parser.add_argument( "-f", "--first",   default=False, action="store_true" )
        parser.add_argument( "-x", "--index",  type=int, default=-1 )
        parser.add_argument( "-c", "--count",  action="store_true", default=False )
        parser.add_argument( "-m", "--mode",  choices=['pc', 'cmake', 'cf', 'compare', 'cmake_direct'], default=default_mode  )


        parser.add_argument( "--casesensitive",  action="store_true", default=False )
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        args.lnames = map(lambda _:_.lower(), args.names)
        if args.first:
           args.index = 0 
        pass
        log.debug(args)
        return args 

    def __init__(self, bases, args):
        ubases = []  
        for base in bases:
            if not base in ubases:
                ubases.append(base)
            pass   
        pass
        self.bases = ubases
        self.args = args
        self.pkgs = []
        self.find_config()


    def select(self, args):
        if len(args.names) > 0 and args.casesensitive:
            pkgs = filter(lambda pkg:pkg.name in args.names, self.pkgs)
        elif len(args.lnames) > 0:
            pkgs = filter(lambda pkg:pkg.name.lower() in args.lnames, self.pkgs)
        else:
            pkgs = self.pkgs
        pass
        return pkgs[args.index:args.index+1] if args.index > -1 else pkgs




class FindPkgConfigPkgs(Find):

    CONFIG = re.compile("(?P<pcname>\S*?).pc$")

    def __init__(self, bases, args):
        Find.__init__(self, bases, args)

    def find_config(self):
        for base in self.bases:  
            if not os.path.isdir(base):
                log.debug("base %s does not exist " % base)
            else:    
                self.find_config_(base)
            pass
        pass   

    def find_config_(self, base):
        """
        :param base: directory in which to look for pc files
        """
        log.info("find_config_ %s " % (base)) 
        names = os.listdir(base)
        for name in names:
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                m = self.CONFIG.match(name)
                if not m: continue
                pcname = m.groupdict()['pcname']  
                pkg = Pkg.CreateFromPC(path)
                assert pkg.name == pcname, (pkg.name, pcname) 

                self.pkgs.append(pkg) 
            pass
        pass



class FindCMakePkgDirect(Find):
    """ 
    FindCMakePkgDirect
    -------------------

    Uses CMake directly to find packages and collect metadata by parsing CMake output.
    Implemented with a CMakeLists.txt and go.sh script generated into a temporary directory.

    * https://cmake.org/cmake/help/latest/command/find_package.html

    How to know if a pkg needs should be found via MODULE or CONFIG ?
       
         * cmake tries first MODULE-wise FindName.cmame in the CMAKE_MODULE_PATH 
           but then looks CONFIG-wise in the CMAKE_PREFIX_PATH for 
           lib/cmake/lowercasename-config.cmake or lib/cmake/NameConfig.cmake 

         * NB but find_package still needs the correctly cased name argument

    Related investigations
    ~~~~~~~~~~~~~~~~~~~~~~~~

    cmake/Modules/OpticksCMakeTraverse.cmake
         traversing all targets from the highest level G4OK on down in 
         a single CMake invokation offers a way to speed this up dramatically 
         and in pricipal to implement an opticks-config with no need for pc files
         and pkg-config 

    TODO
    ~~~~~

    * try to simplify metadata extraction, currently using a mess of different properties
      and variable dumping : whereas there should be a clean way of doing this 

    """
    PTN = re.compile("^-- (?P<var>\S*)=(?P<val>.*)\s*$")
    FIND = re.compile("^Find(?P<pkg>\S*)\.cmake$") 
    NAME = re.compile("# PROJECT_NAME\s*(?P<name>\S*)\s*$")

    def __init__(self, bases, args):
        Find.__init__(self, bases, args)

    def find_config_names(self):
        """
        :return pkgnames: glob finds CONFIG-wise lib/cmake/*/*-config.cmake 

        The lowecasename-config.cmake files are grepped for PROJECT_NAME  
        to get the original non-lowercased names.  The PROJECT_NAMES
        comments are planted by BCM export.
        """
        paths = []
        for prefix in os.environ["CMAKE_PREFIX_PATH"].split(":"):
            ppaths = glob.glob("%s/lib/cmake/*/*-config.cmake" % prefix)
            paths += ppaths
            log.debug(" ppaths %2d prefix %s " % (len(ppaths), prefix))
        pass

        names = []
        for path in paths:
            cmd = "cat %s | grep ^#\ PROJECT_NAME -" % path 
            rc,out = getstatusoutput(cmd)
            name = None
            for line in out.split("\n"):
                m = self.NAME.match(line)
                if not m: continue
                name = m.groupdict()["name"]
            if not name is None:
                log.debug("parsed name %s from path %s " % (name, path))
                names.append(name)
            else:
                log.debug("failed to parse name from path %s " % path )
            pass
        pass
        return names

    def find_other_config_names(self): 
        """
        :return pkgnames: os.walk finds '*Config.cmake' beneath $OPTICKS_PREFIX/externals/lib 
        """
        skip_other_config_names=['glfw3', 'GLFW3', 'Geant4']  # these come in under other names: OpticksGLFW, G4 
        names = []
        for root, dirnames, filenames in os.walk(os.path.expandvars("$OPTICKS_PREFIX/externals/lib")):
            fnames = fnmatch.filter(filenames, '*Config.cmake')
            if len(fnames) == 0: continue 
            #print(" root:%s %d " % (root,len(fnames)))
            for filename in fnames:
                name = filename.replace("Config.cmake","")
                #print("%s : %s " % (filename, name))
                if name in skip_other_config_names:
                    log.debug("skipping name %s " % name)
                else:
                    names.append(name)
            pass
        pass
        #print("names : %s " % repr(names))
        return names

    def find_module_names(self):
        """
        :return pkgnames: list of MODULE package names of form FindName.cmake from cmake/Modules
        """
        names = []
        path = os.path.expandvars("$OPTICKS_HOME/cmake/Modules") 
        # not installed tree OPTICKS_PREFIX as this should work before Opticks installation
        for name in os.listdir(path):
            m = self.FIND.match(name)
            if not m: continue
            pkg = m.groupdict()["pkg"] 
            names.append(pkg)
        pass
        return names 

    def find_config(self):
        pass
        if len(self.args.names) == 0:
            self.module_names = self.find_module_names()
            self.config_names = self.find_config_names()
            self.other_config_names = self.find_other_config_names()
        else:
            self.module_names = []
            self.config_names = []
            self.other_config_names = []
        pass
        self.all_names = self.module_names + self.config_names + self.other_config_names

        if self.args.dumpnames: 
            self.dump_names()
        pass

        find_names = self.all_names if len(self.args.names)==0 else self.args.names

        for name in find_names:
            pkg = self.find_package(name)
            if pkg["rc"] != 0:
                print(" -------------- %s ------------- " % name )
                print("FAILED TO FIND %s " % name)
                #print(pkg["out"])
            else:
                if self.args.dump:
                    print(" -------------- %s ------------- " % name )
                    print(pkg["out"])
                    print(str(pkg))
                pass
                pass
                self.pkgs.append(pkg)
            pass
        pass

    def dump_names(self):
        print("\n".join(["---module_names---"]+self.module_names))
        print("\n".join(["---config_names---"]+self.config_names))
        print("\n".join(["---other_config_names---"]+self.other_config_names))

    def find_package(self, name):
        """
        Finding is a bit slow so cache them
        """
        log.info("find_package %s " % name)
        cache_path = DirectPkg.Path(name)
        cache_exists = DirectPkg.Exists(name)
        cache = self.args.cache 
        log.debug("find_package.name:%s cache_exists:%s cache:%s cache_path:%s  " % (name,cache_exists,cache, cache_path) )

        if cache_exists and cache:
            d = DirectPkg.Load(name)
        else: 
            d = self.find_package_(name)
            DirectPkg.Save(d)
            pass
        pass
        return DirectPkg(d)

    def find_package_(self, name, opts="REQUIRED"): 
        """
        :param name: case sensitive name, eg OpticksGLFW OpticksGLEW OptiXRap
        """
        rc = 0 
        d = dict()
        d["name"] = name
        d["args"] = vars(self.args)  # cannot json serialize a Namespace

        log.debug("find_package_.name %s " % name )

        with TemporaryDirectory() as tmpdir:

            log.debug("find_package_.tmpdir %s " % tmpdir)

            cm = CMakeLists(pkg=name, opts=opts)
            go = Script()
            file("CMakeLists.txt", "w").write(str(cm))

            sh = "./go.sh"
            file(sh, "w").write(str(go))
            mode = os.stat(sh).st_mode
            mode |= (mode & 0o444) >> 2   
            os.chmod(sh, mode)

            rc,out = getstatusoutput(sh)

            log.debug("RC %d : dumping out between hashes\n#####################\n%s\n###############\n" % (rc,out) )

        pass

        d["rc"] = rc 
        d["out"] = out 

        if rc == 0:
            for line in out.split("\n"):
                m = self.PTN.match(line) 
                if not m:continue
                var = m.groupdict()["var"]
                val = m.groupdict()["val"]
                d[var] = val 
            pass
        pass
        return d



class FindCMakePkgs(Find):
    """
    This is the simple heuristic finder that only works CONFIG-wise 
    """

    CONFIG = re.compile("(?P<pfx>\S*?)-?[cC]onfig.cmake$")

    PRUNE = ["Modules", "Linux-g++", "Darwin-clang"]  # Linux-g++ and Darwin-clang are .. symbolic uplinks that cause infinite recursion

    BUILD = re.compile(".*build.*")

    SUBS = ["lib", "lib64"]


    def __init__(self, bases, args):
        Find.__init__(self, bases, args)

    def find_config(self):
        """
        Constructs a list of unique existing libdirs from all the bases
        then invokes find_config_r for each of them    
        """
        vbases = []
        for base in self.bases:
            for sub in self.SUBS:
                path = os.path.join(*filter(None,[base, sub])) 
                if not os.path.isdir(path): continue
                vbases.append(path)  
            pass
        pass
        for base in vbases:  
            self.find_config_r(base,0)
        pass   

    def find_config_r(self, base, depth):
        """
        :param base: directory
        :param depth: integer

        Recursively traverses a directory heirarchy steered 
        by directory names to skip in self.PRUNE and self.BUILD.
        For each file name matching self.CONFIG creates and collects
        a Pkg instance.

        Note there is no selection at this stage all pkgs 
        are collected.
        """
        log.debug("find_config_r %2d %s " % (depth, base)) 
        names = os.listdir(base)
        for name in names:
            path = os.path.join(base, name)
            if os.path.isdir(path):
                if name in self.PRUNE:
                    pass
                else:
                    m = self.BUILD.match(name) 
                    if m:
                        log.debug("build match %s " % name)
                    else:
                        self.find_config_r(path, depth+1)
                    pass
            else:
                m = self.CONFIG.match(name)
                if not m: continue
                pfx = m.groupdict()['pfx']  
                if len(pfx) == 0 or pfx == "CTest" or pfx.startswith("BCM") or pfx.find("Targets-") > -1: continue

                pkg = Pkg(path, pfx)
                self.pkgs.append(pkg) 
            pass
        pass




class Main(object):
    def get_bases(self, var):
        pp = os.environ.get(var,"")
        bases = filter(None, pp.split(":"))
        log.debug("\n".join(["%s:" % var]+bases)) 
        return bases

    def find_pc_pkgs(self):
        args = self.args
        pc_bases = self.get_bases("PKG_CONFIG_PATH")
        fpc = FindPkgConfigPkgs(pc_bases, args)
        pc_pkgs = fpc.select(args)
        return pc_pkgs

    def find_cm_pkgs(self): 
        """
        Formerly used CONFIG-wise approach, missing out on MODULE finds, with::

           #cm_bases = self.get_bases("CMAKE_PREFIX_PATH")
           #fcm = FindCMakePkgs(cm_bases, args)
         
        """
        args = self.args
        fcm = FindCMakePkgDirect([], args)  
        cm_pkgs = fcm.select(args)
        return cm_pkgs

    def __init__(self, default_mode="cmake"):

        args = Find.parse_args(__doc__, default_mode=default_mode)
        self.args = args

        rc = 0 
        if args.mode in ["pc", "cmake"]:
            pkgs = []
            if args.mode == "pc":
                pkgs = self.find_pc_pkgs()
            elif args.mode == "cmake":
                pkgs = self.find_cm_pkgs()
            else:
                assert 0 
            pass
            log.debug("[dumping %d pkgs " % len(pkgs))
            self.dump(pkgs)
            log.debug("]dumping %d pkgs " % len(pkgs))
        elif args.mode in ["cf", "compare"]:

            log.info("find_cm_pkgs")  
            cm_pkgs = self.find_cm_pkgs()
            log.info("find_pm_pkgs")  
            pc_pkgs = self.find_pc_pkgs()

            print("--- CMake")
            self.dump(cm_pkgs)
            print("--- pkg-config")
            self.dump(pc_pkgs)
            cf = Compare("cm",cm_pkgs, "pc",pc_pkgs) 
            print("compare : %s " % repr(cf))
        pass
        sys.exit(rc)


    def dump(self, pkgs):
        args = self.args
        count = len(pkgs)

        if args.count:
            print(count)
        else:    
            for pkg in pkgs:
                if args.libdir:
                    print(pkg.libdir)
                elif args.includedir:
                    print(pkg.includedir)
                elif args.prefix:
                    print(pkg.prefix)
                elif args.name:
                    print(pkg.name)
                else:
                    print(repr(pkg))
                pass
            pass
        pass



if __name__ == '__main__':
    Main(default_mode="compare")
  


