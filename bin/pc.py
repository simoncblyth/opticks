#!/usr/bin/env python

import os, re, argparse
from collections import OrderedDict as odict



class Line(object):
    varptn = re.compile("^(?P<key>\S*)=(?P<val>\S*)\s*$")
    def __init__(self, i_line):
        i, line = i_line
        self.l = line
        self.i = i  
        m = self.varptn.match(line)
        if m:
            d = m.groupdict()
            k = d["key"]
            v = d["val"]
        else:
            k = ""
            v = ""
        pass
        self.k = k 
        self.v = v 

    def __str__(self):
        return "%2d : %-80s : %-15s : %-20s " % ( self.i, self.l, self.k, self.v )


class PC(object):
    """

    Canonically invoked from oc-pcfix

    Parse variable lines from PC files of the below form:: 

        prefix=/usr/local/opticks
        exec_prefix=${prefix}/externals
        includedir=${prefix}/externals/include
        libdir=${exec_prefix}/lib

    """

    @classmethod
    def Find(cls, name):
        """
        """
        pcp = os.environ.get("PKG_CONFIG_PATH","")
        dirs = filter(None,pcp.split(":"))
        for d in dirs:
            path = os.path.join(d, "%s.pc" % name.lower() )
            if os.path.exists(path):
                return cls(path)
            pass
        return None

    def __init__(self, path):
        pass
        self.path = path 
        self.d = odict()
        self.pc = self.parse( map(str.strip, file(path).readlines()) ) 
       
    def parse(self, lines):
        self.ls = map(Line, enumerate(lines))
  
    def __repr__(self):
        return "PC %s " % self.path 

    def __str__(self):
        return "\n".join(map(str, self.ls))


class Main(object):
    @classmethod
    def Parse(cls):
        parser = argparse.ArgumentParser(__doc__)
        parser.add_argument( "name", nargs='*', help="Name of pc file(s) eg glfw3,assimp,glew " )
        parser.add_argument( "--fix" , action="store_true", default=False )
        parser.add_argument( "--prefix", type=str,    default="/usr/local/opticks" )
        parser.add_argument( "--includedir", type=str,default="${prefix}/externals/include" )
        parser.add_argument( "--libdir", type=str,    default="${prefix}/externals/lib" )
        args = parser.parse_args()
        return args

    def __init__(self):
        self.args = self.Parse() 
        print(self.args)
        for name in self.args.name:
            pc = PC.Find(name)
            print(pc)
        pass

if __name__ == '__main__':
    main = Main()

