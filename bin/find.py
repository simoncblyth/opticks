#!/usr/bin/env python
"""
find.py
========

Ask CMake directly what it finds instead of using heuristic. 

* https://cmake.org/cmake/help/latest/command/find_package.html

How to know if a pkg needs should be found via MODULE or CONFIG ?
   
     * cmake tries first MODULE-wise FindName.cmame in the CMAKE_MODULE_PATH 
       but then looks CONFIG-wise in the CMAKE_PREFIX_PATH for 
       lib/cmake/lowercasename-config.cmake or lib/cmake/NameConfig.cmake 

     * NB but find_package still needs the correctly cased name argument


The below are problematic, as they dont follow any standard::

    epsilon:opticks blyth$ find externals/lib -name '*Config.cmake'
    externals/lib/cmake/Boost-1.70.0/BoostConfig.cmake
    externals/lib/cmake/glfw/glfw3Config.cmake
    externals/lib/Geant4-10.4.2/Geant4Config.cmake

CMake 3.17.1 even failing to find that glfw3::

    ## content of the config uses GLFW3_ prefix so need to use that 
    ## for the filenames and dirnames for modern CMake to find it CONFIG-wise

    ## rename via temps for case-insensitive macOS
    epsilon:glfw blyth$ mv glfw3Config.cmake glfw3Config.cmake.tmp
    epsilon:glfw blyth$ mv glfw3Config.cmake.tmp GLFW3Config.cmake
    epsilon:glfw blyth$ mv glfw3ConfigVersion.cmake glfw3ConfigVersion.cmake.tmp
    epsilon:glfw blyth$ mv glfw3ConfigVersion.cmake.tmp GLFW3ConfigVersion.cmake
    epsilon:cmake blyth$ mv glfw GLFW3

    ## ahha : cmake/Modules/FindOpticksGLFW.cmake is a workaround for this problem 
    ## so can just exclude the GLFW3 name 


"""
import os, shutil, tempfile, commands, stat, re, glob, logging, fnmatch
from collections import OrderedDict as odict
from findpkg import Find, getlibdir

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
        return self.TMPL % self
 
class CMakeLists(Tmpl):
    TMPL=r"""
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
    TMPL=r"""#!/bin/bash 

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
    -DCMAKE_MODULE_PATH=$OPTICKS_PREFIX/cmake/Modules \
    -DOPTICKS_PREFIX=$OPTICKS_PREFIX

"""



class DirectPkg(odict): 
    def __str__(self):
        return "\n".join(["%-40s = %s" % tuple(kv) for kv in self.items()])

    def __repr__(self):
        return "%-20s : %-60s : %s  " % (self.name, self.includedir, self.libdir )  

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
        """
        first prefix encountered that the includedir startswith
        """
        incdir = self.includedir
        for prefix in os.environ.get("CMAKE_PREFIX_PATH","").split(":"):
            if incdir.startswith(prefix):
                return prefix
            pass
        return "" 

    def _get_includedir(self):
        return self.mget(["INCLUDE_DIR", "INTERFACE_INCLUDE_DIRECTORIES"])

    prefix = property(_get_prefix)
    libdir = property(_get_libdir)
    includedir = property(_get_includedir)



class FindCMakePkgDirect(Find):
    """ 
    """

    PTN = re.compile("^-- (?P<var>\S*)=(?P<val>.*)\s*$")
    FIND = re.compile("^Find(?P<pkg>\S*)\.cmake$") 
    NAME = re.compile("# PROJECT_NAME\s*(?P<name>\S*)\s*$")

    def find_config_names(self):
        """
        find . -name '*-config.cmake'  -exec grep -H PROJECT_NAME {} \;
        find $OPTICKS_PREFIX/externals/lib -name '*Config.cmake'
        """
        paths = glob.glob(os.path.expandvars("$OPTICKS_PREFIX/lib/cmake/*/*-config.cmake"))
        names = []
        for path in paths:
            cmd = "cat %s | grep ^#\ PROJECT_NAME -" % path 
            rc,out = commands.getstatusoutput(cmd)
            for line in out.split("\n"):
                m = self.NAME.match(line)
                if not m: continue
                name = m.groupdict()["name"]
                names.append(name)
            pass
        pass
        return names

    def find_other_config_names(self): 
        skip_other_config_names=['glfw3', 'GLFW3']
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
        path = os.path.expandvars("$OPTICKS_PREFIX/cmake/Modules")
        for name in os.listdir(path):
            m = self.FIND.match(name)
            if not m: continue
            pkg = m.groupdict()["pkg"] 
            names.append(pkg)
        pass
        return names 


    def find_config(self):
        pass
        self.module_names = self.find_module_names()
        self.config_names = self.find_config_names()
        self.other_config_names = self.find_other_config_names()
        #print("\n".join(pkgnames))

    def find_package(self, name, opts): 
        """
        :param name: case sensitive name, eg OpticksGLFW OpticksGLEW OptiXRap

        Hmm does it make sense to prefix all the MODULEs in 
        cmake/Modules/FindOpticksName.cmake

        OpticksCUDA gives target Opticks::CUDA
        
        ./find.py OpticksCUDA ImGui G4 GLM 

        """
        rc = 0 
        d = {}
        d["name"] = name
        d["args"] = args

        with TemporaryDirectory() as tmpdir:
            cm = CMakeLists(pkg=name, opts=opts)
            go = Script()
            file("CMakeLists.txt", "w").write(str(cm))

            sh = "./go.sh"
            file(sh, "w").write(str(go))
            mode = os.stat(sh).st_mode
            mode |= (mode & 0o444) >> 2   
            os.chmod(sh, mode)

            rc,out = commands.getstatusoutput(sh)
            for line in out.split("\n"):
                #print(" +++ " + line)
                m = self.PTN.match(line) 
                if not m:continue
                var = m.groupdict()["var"]
                val = m.groupdict()["val"]
                d[var] = val 
            pass
            if rc != 0:
                return None, out 
            pass
        pass
        return DirectPkg(d), out

       



if __name__ == '__main__':
    args = Find.parse_args(__doc__, default_mode="cmake_direct")
    #print(args)
    fd = FindCMakePkgDirect([])  

    if args.dump: 
        print("\n".join(["---module_names---"]+fd.module_names))
        print("\n".join(["---config_names---"]+fd.config_names))
        print("\n".join(["---other_config_names---"]+fd.other_config_names))
    pass
    all_names = fd.module_names + fd.config_names + fd.other_config_names

    names = args.names if len(args.names) > 0 else all_names
    for name in names:
        pkg,out = fd.find_package(name, "REQUIRED")
        if pkg == None:
            print(" -------------- %s ------------- " % name )
            print("FAILED TO FIND %s " % name)
            print(out)
        else:
            if args.dump:
                print(" -------------- %s ------------- " % name )
                print(out)
                print(str(pkg))
            pass
            print(repr(pkg))
        pass
    pass




