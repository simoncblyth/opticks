#!/usr/bin/env python
"""


"""
import os, re, sys, argparse, logging
log = logging.getLogger(__name__)
from collections import OrderedDict as odict

class Line(object):
    fixkey = ["prefix","includedir","libdir"]
    varptn = re.compile("^(?P<key>\S*)=(?P<val>\S*)\s*$")

    enum = ["NO-VARFIX","VARFIX","-"]

    def __init__(self, i_line):
        """
        :param i_line: 

        Match key=val lines of the pc file

        """
        i, line = i_line
        self.l = line
        self.i = i  

        m = self.varptn.match(line)
        if m:
            d = m.groupdict()
            k = d["key"]
            v = d["val"]
            if k in self.fixkey:
                fv = self.vargs[k]        ## from commandline inputs or defaults      
                s = 0 if v == fv else 1   ## is a fix needed 
            else:
                fv = v 
                s = 0
            pass
        else:
            k = ""
            v = ""
            s = 2
        pass
        self.k = k 
        self.v = v 
        self.fv = fv if s == 1 else v     ## fixed value 
        self.s = s
        self.e = self.enum[s]


    def __str__(self):
        return "%s=%s" % (self.k, self.fv) if self.s == 1 else self.l

    def __repr__(self):
        return "%2d : %-80s : %-15s : %-20s : %-20s : %s " % ( self.i, self.l, self.k, self.v, self.fv, self.e )


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
    def Resolve(cls, name):
        """
        :param name:
        :return pc: PC instance or None

        Search the PKG_CONFIG_PATH dirs for name.pc where name is lowercased. 
        unless name looks like an absolute path.
        """
        if name[0] == "/" and os.path.exists(name):
            path = name
            return path
        pass 
        pcp = os.environ.get("PKG_CONFIG_PATH","")
        dirs = filter(None,pcp.split(":"))
        for d in dirs:
            #path = os.path.join(d, "%s.pc" % name.lower() )
            path = os.path.join(d, "%s.pc" % name )
            if os.path.exists(path):
                return path
            pass
        pass
        assert 0
        return None

    @classmethod
    def Find(cls, name):
        path = cls.Resolve(name)
        if path is None:
            log.fatal("failed to find %s in any PKG_CONFIG_PATH " % (name)) 
            return None
        pass
        return cls(path) 


    def __init__(self, path):
        """
        :param path: to a pc file
     
        Read the lines of a pc file
        """
        pass
        self.path = path 
        lines =  map(str.strip, file(path).readlines())
        self.ls = map(Line, enumerate(lines))
  
    def _get_numfix(self):
        """
        count the number of fixes
        """
        nf = 0 
        for l in self.ls:
            if l.s == 1: nf += 1 
        pass
        return nf 
    numfix = property(_get_numfix)    

    def _get_header(self):
        return "# %s numfix:%d %s " % (sys.argv[0], self.numfix, self.path) 
    header = property(_get_header)

    def __repr__(self):
        return "\n".join([self.header, ""] + map(repr, self.ls))
    def __str__(self):
        return "\n".join([self.header, ""] + map(str, self.ls))


class Main(object):
    @classmethod
    def Parse(cls):
        parser = argparse.ArgumentParser(__doc__)
        parser.add_argument( "name", nargs='*', help="Name of pc file(s) eg glfw3,assimp,glew " )
        parser.add_argument( "--fix" , action="store_true", default=False )

        default_prefix = os.environ.get("OPTICKS_PREFIX","/usr/local/opticks")
        default_includedir = "${prefix}/externals/include" 
        default_libdir = "${prefix}/externals/lib"

        parser.add_argument( "--prefix", type=str,    default=default_prefix )
        parser.add_argument( "--includedir", type=str,default=default_includedir )
        parser.add_argument( "--libdir", type=str,    default=default_libdir )
        args = parser.parse_args()
        return args

    def __init__(self):
        """
        Hmm the appropriate variables should depend on 
        the location of the found pc file.  

        With some exceptions, eg for optix, cuda which 
        could be avoided. 
        """
        args = self.Parse() 
        Line.vargs = vars(args)
        self.args = args
        for name in self.args.name:
            pc = PC.Find(name)
            print("-" * 100)
            print(str(pc))
            print("-" * 100)
            print(repr(pc))
            print("-" * 100)
            if args.fix and pc.numfix > 0:
                open(pc.path, "w").write(str(pc))
                log.info("writing fixed pc %s " % pc.path )
            pass
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main = Main()

