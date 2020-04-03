#!/usr/bin/env python

"""


"""

import argparse

class Config(object):
    """ 
    Config
    -------

    Classmethods are used to access libs and flags while object 
    methods handle dependencies and recursing over those. 

    """
    @classmethod
    def Get(cls, pkg):
        default = cls.D.get("DEFAULT", "") 
        s = cls.D.get(pkg, default) % { 'pkg':pkg }
        if s == "":
            return ""
        pass 
        ipfx = cls.D.get("ITEM_PREFIX", "")
        return "%s%s" % (ipfx,s) 

    @classmethod
    def Lines(cls, deps):
        return [cls.D.get("HEAD_PREFIX","")] + map(lambda pkg:cls.Get(pkg), deps)


    def __init__(self):
        self.dd = []

    def add(self, p):
        do_add = not p in self.dd
        if do_add:
            self.dd.append(p)
        pass
        return do_add

    def direct(self, pkg):
        return self.D.get(pkg,[])

    def recurse(self, pkg):
        self.dd = []
        def all_r(p):
            ddeps = self.direct(p)  
            for ddep in ddeps:
                do_add = self.add(ddep)
                if do_add:
                    all_r(ddep)
                pass
            pass
        pass
        all_r(pkg)
        return self.dd



class Libs(Config):
    D = {
      "ITEM_PREFIX":"-l",
      "HEAD_PREFIX":"-L%(opticksprefix)s/lib",
      "DEFAULT":"%(pkg)s", 

      "CPP":"stdc++",
      "PLog":"",
      "GLM":"",
      "OpticksAssimp":"",
    } 

class Flags(Config):
    D = {
       "ITEM_PREFIX":"-I",
       "HEAD_PREFIX":"",
       "DEFAULT":"%%(opticksprefix)s/include/%(pkg)s",

       "CPP":"",
       "PLog":"%%(opticksprefix)s/externals/plog/include",
       "GLM":"%%(opticksprefix)s/externals/glm/glm",
       "OpticksAssimp":"",
    }

class Context(Config):
    D = {
           "libdir":"%%(opticksprefix)s/lib", 
           "extdir":"%%(opticksprefix)s/externals", 
        }


class Deps(Config):
    D = {
      "CPP":[], 
      "PLog":[], 
      "GLM":[],
      "SysRap":["PLog"],
      "BoostRap":["SysRap", "CPP"],
      "NPY":["BoostRap","GLM"],
      "YoctoGLRap":["NPY"],
      "OpticksCore":["NPY"],
      "GGeo":["OpticksCore","YoctoGLRap"],
      "AssimpRap":["OpticksAssimp", "GGeo"],  
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(  "pkg", nargs='*', help="Package" )
    parser.add_argument("-v","--verbose", action="store_true", default=False )
    parser.add_argument("--direct", action="store_true", default=False )

    parser.add_argument("-l","--libs", action="store_true", default=False )
    parser.add_argument("-f","--flags", action="store_true", default=False )
    parser.add_argument("-d","--deps", action="store_true", default=False )
    parser.add_argument("--libdir", action="store_true", default=False )
    parser.add_argument("--libpath", action="store_true", default=False )

    parser.add_argument("--prefix", default="/usr/local/opticks" )
    parser.add_argument("--raw", action="store_true", default=False )


    args = parser.parse_args()
    pkg = args.pkg[0] if len(args.pkg) == 1 else None

    ctx = {'opticksprefix':args.prefix }

    dp = Deps()

    if pkg is None:
        deps = None
    else:
        deps = dp.direct(pkg) if args.direct else dp.recurse(pkg)
    pass

    if args.deps:
        ret = deps
    elif args.libs:
        ret = Libs.Lines([pkg]+deps)
    elif args.flags:
        ret = Flags.Lines([pkg]+deps)
    elif args.libdir or args.libpath:
        ret = [Context.Get("libdir")]
    else:
        ret = deps
    pass
    
    if args.raw:
        iret = ret
    else:
        iret = map( lambda s:s % ctx, ret )
    pass

    delim = ":" if args.libpath else "\n" 


    print(delim.join(filter(None,iret)))

