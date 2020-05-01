#!/usr/bin/env python
"""
find.py
========

Ask CMake directly what it finds instead of using heuristic. 


How to know if a pkg needs should be found via MODULE or CONFIG ?
   look at the FindName.cmake 


"""
import os, shutil, tempfile, commands, stat 

class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp() so it's usable with "with" statement."""
    def __enter__(self):
        self.iwd = os.getcwd()

        path = tempfile.mkdtemp()

        #path = "/tmp/tt"
        #os.makedirs(path)

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

set(%(pkg)s_VERBOSE ON) 

find_package(%(pkg)s %(args)s)
"""
          
class Script(Tmpl):
    TMPL=r"""#!/bin/bash 

pwd
ls -l

echo OPTICKS_PREFIX $OPTICKS_PREFIX
echo CMAKE_PREFIX_PATH 
echo $CMAKE_PREFIX_PATH | tr ":" "\n"

mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
    -DCMAKE_MODULE_PATH=$OPTICKS_PREFIX/cmake/Modules \
    -DOPTICKS_PREFIX=$OPTICKS_PREFIX

"""


class Parse(object):
    """
    -- FindPLog.cmake : PLog_INCLUDE_DIR : /usr/local/opticks/externals/plog/include 
    """

if __name__ == '__main__':
  
    with TemporaryDirectory() as tmpdir:
        print(tmpdir)
        cm = CMakeLists(pkg="PLog", args="REQUIRED MODULE")
        go = Script()
        file("CMakeLists.txt", "w").write(str(cm))

        sh = "./go.sh"
        file(sh, "w").write(str(go))
        mode = os.stat(sh).st_mode
        mode |= (mode & 0o444) >> 2   
        os.chmod(sh, mode)

        st,out = commands.getstatusoutput(sh)
        print(st)

        for line in out.split("\n"):
            print(" +++ " + line)
        pass
    pass
    


