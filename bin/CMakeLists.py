#!/usr/bin/env python

import sys, re, os

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

   def __init__(self, lines):
       self.lines = lines 
       self.name = None
       self.deps = []
       self.parse()
  
   def parse(self):
       for line in self.lines:
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

   def __repr__(self):
       return "%20s : %s " % (  self.name, " ".join(map(lambda _:_.name, self.deps)) )

   def __str__(self):
       return "\n".join(self.lines)  


class Opticks(object):

    order = {
             'OKConf':10, 
             'SysRap':20, 
             'BoostRap':30, 
             'NPY':40, 
             'OpticksCore':50, 
             'GGeo':60, 
             'AssimpRap':70,
             'OpenMeshRap':80, 
             'OpticksGeometry':90,
             'CUDARap':100,
             'ThrustRap':110,
             'OptiXRap':120,
             'OKOP':130,
             'OGLRap':140,
             'OpticksGL':150,
             'OK':160,
             'CFG4':170,
             'okg4':180,
             'NumpyServer':-1
            }

    @classmethod
    def examine_dependencies(cls):
        root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print root    
        pkgs = {} 
        for dirpath, dirs, names in os.walk(root):
            if CMakeLists.NAME in names and not "examples" in dirpath and not "tests" in dirpath and not "externals" in dirpath:
                path = os.path.join(dirpath, CMakeLists.NAME)
                lines = map(str.strip, file(path,"r").readlines() ) 
                ls = CMakeLists(lines)
                pkgs[ls.name] = ls
                #print path
                #print repr(ls)
            pass
        pass
       
        for k in sorted(filter(lambda k:cls.order.get(k) > -1,pkgs.keys()), key=lambda k:cls.order.get(k,1000)):
            print "%3s %s " % ( cls.order.get(k,1000), repr(pkgs[k]) )
        pass


    def readstdin(self):
        lines = map(str.strip, sys.stdin.readlines() ) 
        ls = CMakeLists(lines)
        #print ls 
        print repr(ls)
    pass


if __name__ == '__main__':
    Opticks.examine_dependencies()

