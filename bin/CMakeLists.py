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

from __future__ import print_function
import sys, re, os, logging, argparse
log = logging.getLogger(__name__)

class Dependent(object):
    def __init__(self, findargs):
        findargs = findargs.split()
        self.findargs = findargs
        self.name = findargs[0]

    def __str__(self):
        return "Dependent %s " % " ".join(self.findargs)

    def __repr__(self):
        return "Dependent %s " % self.name

class CMakeLists(object):
   """  
   """
   NAME = "CMakeLists.txt"
   name_ptn = re.compile("^set\(name (?P<name>\S*)\).*")
   find_ptn = re.compile("^find_package\((?P<findargs>.*)\).*")
   obo_txt = "include(OpticksBuildOptions)"

   @classmethod
   def HasOpticksBuildOptions(cls, lines): 
       obo_found = False
       for line in lines:
           if line.startswith(cls.obo_txt):
               obo_found = True 
           pass
       return obo_found

   def __init__(self, lines, reldir=None, path=None, tag=None, precursor=None):
       self.lines = lines 
       self.reldir = reldir
       self.path = path
       self.tag = tag
       self.precursor = precursor
       self.name = None
       self.deps = []
       self.parse()
  
   def parse(self):
       """
       Parse lines from a single CMakeList.txt
       """
       obo_found = False
       for line in self.lines:
           if line.startswith(self.obo_txt):
               obo_found = True 
           pass 
           name_match = self.name_ptn.match(line)
           find_match = self.find_ptn.match(line)
           if name_match:
               self.name = name_match.groupdict()['name']
           elif find_match:
               findargs = find_match.groupdict()['findargs']  
               self.deps.append(Dependent(findargs))
           else:
               pass
           pass
       pass

       if not obo_found:
           log.debug("no obo %s " % self.path)
       pass
       assert obo_found, "missing obo for %s " % self.reldir  


   FMT = "%13s : %13s : %13s : %13s : %s "
   @classmethod
   def columns(cls):
       return cls.FMT % ( "API_TAG", "reldir", "bash-", "Proj.name", "dep Proj.names" ) 

   def __repr__(self):
       return self.FMT  % (  self.tag, self.reldir, self.precursor, self.name, " ".join(map(lambda _:_.name, self.deps)) )

   def _get_tree(self):
       return "\n".join([self.name] + map(lambda _:"    %s" % _.name, self.deps))
   tree = property(_get_tree)

   def __str__(self):
       return "\n".join(self.lines)  


class OpticksCMakeProj(object):
    """
    NB the order keys must correspond to the names as defined by the 
    line in the CMakeLists.txt of form::

       set(name Integration)  

    """
    order = {
             'OKConf':10, 
             'SysRap':20, 
             'BoostRap':30, 
             'NPY':40, 
             'YoctoGLRap':45,
             'OpticksCore':50, 
             'GGeo':60, 
             'AssimpRap':-70,
             'OpenMeshRap':80, 
             'OpticksGeo':90,
             'CUDARap':100,
             'ThrustRap':110,
             'OptiXRap':120,
             'OKOP':130,
             'OGLRap':140,
             'OpticksGL':150,
             'OK':160,
             'ExtG4':165,
             'CFG4':170,
             'OKG4':180,
             'G4OK':190,
             'Integration':200,
             'NumpyServer':-1
            }


    @classmethod
    def find_export_tag(cls, names):
        tail = "_API_EXPORT.hh"
        names = list(filter(lambda _:_.endswith(tail), names))
        tag = names[0].replace(tail,"") if len(names) == 1 else None
        return tag 

    @classmethod
    def find_bash_precursor(cls, names):
        tail = ".bash" 
        names = list(filter(lambda _:_.endswith(tail), names))
        stems = list(map(lambda _:_[:-len(tail)], names))
        stems = list(filter(lambda _:not _.endswith("dev"),stems)) 
        stems = list(filter(lambda _:not _.endswith("x4gen"),stems)) 
        precursor = stems[0] if len(stems) == 1 else None
        return precursor


    @classmethod
    def read_pkgs(cls, home=None):
        if home is None:
            home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pass
        log.info("home %s " % home)    
        pkgs = {} 
        for dirpath, dirs, names in os.walk(home):
            if CMakeLists.NAME in names:
                log.debug("proceed %s " % dirpath ) 
                reldir = dirpath[len(home)+1:]
                path = os.path.join(dirpath, CMakeLists.NAME)
                tag = cls.find_export_tag(names)
                precursor = cls.find_bash_precursor(names)
                lines = list(map(str.strip, open(path,"r").readlines() ))

                has_obo = CMakeLists.HasOpticksBuildOptions(lines)
                if not has_obo:
                    log.debug("skipping %s as does not have OpticksBuildOptions" % path )
                    continue
                pass

                ls = CMakeLists(lines, reldir=reldir, path=path, tag=tag, precursor=precursor)
                pkgs[ls.name] = ls
                log.debug(repr(ls))
            else:
                log.debug("skip %s " % dirpath)
            pass
        pass
        return pkgs


    def __init__(self, home=None):
        pkgs = self.read_pkgs(home=home)
        keys = pkgs.keys()
        log.debug(repr(keys))
        for k in keys:
            o = int(self.order.get(k,-2))
            log.debug(" o %3d k %s  " % (o,k))
        pass     
        self.pkgs = pkgs
        self.keys = sorted(filter(lambda k:self.order.get(k,-2) > -1,keys), key=lambda k:self.order.get(k,1000))

    def get(self, q):
        out = [] 
        for k in self.keys:
            ls = self.pkgs[k]
            out.append(getattr(ls,q))
        pass       
        return out 
    subdirs = property(lambda self:self.get('reldir'))   

    def write_testfile(self, path=None):
        fp = sys.stdout if path is None else file(path, "w") 
        print("# Generated by %s " % sys.argv[0], file=fp)
        print("#", file=fp)
        print("# Outputs to stdout the form of a toplevel CTestTestfile.cmake ", file=fp)
        print("# Allows proj-by-proj build ctest dirs to be run all together, just like integrated  ", file=fp)
        print("#", file=fp)
        print("# Usage example:: ", file=fp)
        print("#", file=fp)
        print("#    opticks-deps --testfile 1> $(opticks-bdir)/CTestTestfile.cmake ", file=fp)
        print("#", file=fp)
        print("#", file=fp)
        subs = map(lambda _:"subdirs(\"%s\")" % _, self.subdirs )
        print("\n".join(subs + [""]), file=fp )


    def __call__(self, args):
        if args.dump:
            print("%3s %s " % ("", CMakeLists.columns()))
        pass
        for k in self.keys:
            ls = self.pkgs[k]
            if args.subdirs:
                print(ls.reldir)
            elif args.tags:
                print(ls.tag)
            elif args.subproj:
                print(ls.name)
            elif args.tree:
                print(ls.tree)
            elif args.dump:
                print("%3s %s " % ( self.order.get(k,1000), repr(ls) ))
            else:
                pass
            pass
        pass

    def readstdin(self):
        lines = map(str.strip, sys.stdin.readlines() ) 
        ls = CMakeLists(lines)
        print(repr(ls))
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "--home",  default=None, help="Project home, eg OPTICKS_HOME " )
    parser.add_argument(     "--dump",  action="store_true", default=True, help="Dump CMakeLists repr" )
    parser.add_argument(     "--tree",  action="store_true", help="Dump tree" )
    parser.add_argument(     "--subdirs",  action="store_true", help="Dump just the subdirs" )
    parser.add_argument(     "--tags",  action="store_true", help="Dump just the tags" )
    parser.add_argument(     "--subproj",  action="store_true", help="Dump just the subproj" )
    parser.add_argument(     "--testfile", action="store_true", help="Generate to stdout a CTestTestfile.cmake with all subdirs" ) 
    parser.add_argument(     "--testfilepath", default=None, help="Write testfile to path provided or stdout by default." ) 
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    ok = OpticksCMakeProj(args.home)
    
    if args.testfile:
        ok.write_testfile(args.testfilepath) 
    else: 
        ok(args)
    pass

