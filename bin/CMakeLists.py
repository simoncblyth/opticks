#!/usr/bin/env python

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

   NAME = "CMakeLists.txt"
   name_ptn = re.compile("^set\(name (?P<name>\S*)\).*")
   find_ptn = re.compile("^find_package\((?P<findargs>.*)\).*")
   obo_txt = "include(OpticksBuildOptions)"

   def __init__(self, lines, reldir=None, path=None, tag=None):
       self.lines = lines 
       self.reldir = reldir
       self.path = path
       self.tag = tag
       self.name = None
       self.deps = []
       self.parse()
  
   def parse(self):
       obo_found = False
       for line in self.lines:
           if line.startswith(self.obo_txt):
               obo_found = True 
           pass 
           name_match = self.name_ptn.match(line)
           find_match = self.find_ptn.match(line)
           if name_match:
               name = name_match.groupdict()['name']
               self.name = name
           elif find_match:
               findargs = find_match.groupdict()['findargs']  
               self.deps.append(Dependent(findargs))
           else:
               pass
           pass
       pass

       if not obo_found:
           print self.path
       pass
       assert obo_found, "missing obo for %s " % self.reldir  

   def __repr__(self):
       return "%15s : %20s : %20s : %s " % (  self.tag, self.reldir, self.name, " ".join(map(lambda _:_.name, self.deps)) )

   def _get_tree(self):
       return "\n".join([self.name] + map(lambda _:"    %s" % _.name, self.deps))
   tree = property(_get_tree)

   def __str__(self):
       return "\n".join(self.lines)  


class Opticks(object):
    """
    """

    order = {
             'OKConf':10, 
             'SysRap':20, 
             'BoostRap':30, 
             'NPY':40, 
             'YoctoGLRap':45,
             'OpticksCore':50, 
             'GGeo':60, 
             'AssimpRap':70,
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
             'NumpyServer':-1
            }


    @classmethod
    def find_export_tag(cls, names):
        tail = "_API_EXPORT.hh"
        names = filter(lambda _:_.endswith(tail), names)
        tag = names[0].replace(tail,"") if len(names) == 1 else None
        return tag 

    @classmethod
    def examine_dependencies(cls, args):
        #root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log.info("root %s " % root)    
        pkgs = {} 
        for dirpath, dirs, names in os.walk(root):
            #if CMakeLists.NAME in names and not "examples" in dirpath and not "tests" in dirpath and not "externals" in dirpath:
            if CMakeLists.NAME in names:
                reldir = dirpath[len(root)+1:]
                path = os.path.join(dirpath, CMakeLists.NAME)
                tag = cls.find_export_tag(names)
                lines = map(str.strip, file(path,"r").readlines() ) 
                ls = CMakeLists(lines, reldir=reldir, path=path, tag=tag)
                pkgs[ls.name] = ls
                #print path
                #print repr(ls)
            pass
        pass
       
        if args.testfile:
            print("# Generated by %s " % sys.argv[0])
            print("#")
            print("# Outputs to stdout the form of a toplevel CTestTestfile.cmake ")
            print("# Allows proj-by-proj build ctest dirs to be run all together, just like integrated  ")
            print("#")
            print("# Usage example:: ")
            print("#")
            print("#    opticks-deps --testfile 1> $(opticks-bdir)/CTestTestfile.cmake ")
            print("#")
            print("#")

        for k in sorted(filter(lambda k:cls.order.get(k) > -1,pkgs.keys()), key=lambda k:cls.order.get(k,1000)):

            ls = pkgs[k]
            if args.subdirs:
                print(ls.reldir)
            elif args.tags:
                print(ls.tag)
            elif args.subproj:
                print(ls.name)
            elif args.tree:
                print(ls.tree)
            elif args.dump:
                print(repr(ls))
            elif args.testfile:
                print "subdirs(\"%s\")" % ls.reldir
            else:
                print "%3s %s " % ( cls.order.get(k,1000), repr(ls) )
            pass
        pass


    def readstdin(self):
        lines = map(str.strip, sys.stdin.readlines() ) 
        ls = CMakeLists(lines)
        #print ls 
        print repr(ls)
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "--dump",  action="store_true", help="Dump CMakeLists repr" )
    parser.add_argument(     "--tree",  action="store_true", help="Dump tree" )
    parser.add_argument(     "--subdirs",  action="store_true", help="Dump just the subdirs" )
    parser.add_argument(     "--tags",  action="store_true", help="Dump just the tags" )
    parser.add_argument(     "--subproj",  action="store_true", help="Dump just the subproj" )
    parser.add_argument(     "--testfile", action="store_true", help="Generate to stdout a CTestTestfile.cmake with all subdirs" ) 
    args = parser.parse_args()
    
    Opticks.examine_dependencies(args)

