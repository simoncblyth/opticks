#!/usr/bin/env python
"""
pub.py
=========

Used for example from qudarap/tests/pub.sh::

    #!/bin/bash -l 
    export SRC_BASE=$TMP/QCerenkovTest 
    pub.py $* 

Leading to destination base folder ~/simoncblyth.bitbucket.io/env/presentation/QCerenkovTest/

And CSGOptiX/pub.sh::

    #!/bin/bash -l 
    export SRC_BASE=$TMP/CSGOptiX/CSGOptiXSimulateTest
    pub.py $* 

Leading to destination base folder ~/simoncblyth.bitbucket.io/env/presentation/CSGOptiXSimulateTest/

And extg4/pub.sh::

    #!/bin/bash -l 
    export SRC_BASE=$TMP/extg4/X4IntersectTest
    pub.py $*  

Leading to destination base folder ~/simoncblyth.bitbucket.io/env/presentation/X4IntersectTest/

Usage::

    ./pub.sh sp   #  list source paths : use this to find obsolete ones for deletion 
    ./pub.sh dp   #  list destination paths 
    ./pub.sh xdp   #  list existing destination paths 

    ./pub.sh cp   # emit to stdout the commands to copy png to publication area preserving relative path info
    ./pub.sh cp | sh    # pipe to shell to make copies 
    ./pub.sh cp | grep 1,3 | sh   # select what to copy 


    ./pub.sh s5   # emit to stdout the s5 rst lines to in presentation rst 
    ./pub.sh      # emit to stdout both the above

"""
import os, sys, logging, argparse
log = logging.getLogger(__name__) 

class Pub(object):
    DEST_ROOT = os.path.expanduser("~/simoncblyth.bitbucket.io")
    DEST_BASE = os.path.join(DEST_ROOT, "env/presentation")
    INDENT = "    "

    def __init__(self, base=None, exts=[".png", ".jpg"]):
        if base is None:
            base = os.environ.get("SRC_BASE", "$TMP/NonExisting")
        pass
        expect = os.path.isdir(base)
        if not expect:
            log.fatal("base directory %s does not exist, set envvar SRC_BASE to an existing directory" % base )
        pass
        assert expect
        name = os.path.basename(base)
        dest = os.path.join(self.DEST_BASE, name)

        self.find(base, exts)

        cmds, s5ps, s5ti = self.copy_cmds(dest)

        self.dest = dest
        self.cmds = cmds
        self.s5ps = s5ps
        self.s5ti = s5ti

    def find(self, base, exts):
        """
        Collect base relative paths to files with extension *ext*
        """
        base = os.path.expandvars(base)
        res = []
        for root, dirs, files in os.walk(base):
            for name in files:

                for ext in exts:
                    if name.endswith(ext): 
                        path = os.path.join(root, name)
                        res.append(path[len(base)+1:])
                    pass
                pass
            pass
        pass
        self.base = base
        self.res = sorted(res)

    def copy_cmds(self, dest):
        """
        """
        cmds = []
        s5ps = []
        s5ti = []
        for rel in self.res:
            spath = os.path.join(self.base, rel)
            dpath = os.path.join(dest, rel)
            ddir = os.path.dirname( dpath)
            cmds.append("mkdir -p %s " % ddir)
            cmds.append("cp %s %s " % (spath, dpath))
            s5p = "/"+dpath[len(self.DEST_ROOT)+1:]  
            s5ps.append(s5p)
            elem = rel.split("/")
            title = "_".join([elem[-3], elem[-2], elem[-1]])
            s5ti.append(title)
        pass
        return cmds, s5ps, s5ti

    def spaths(self):
        spl = []
        for rel in self.res:
            spath = os.path.join(self.base, rel)
            spl.append(spath)
        pass
        return spl

    def dpaths(self):
        dest = self.dest 
        dpl = []
        for rel in self.res:
            dpath = os.path.join(dest, rel)
            dpl.append(dpath)
        pass
        return dpl

    def existing_dpaths(self):
        dpl = self.dpaths()
        xdp = []
        for dp in dpl:
            if os.path.exists(dp):
                xdp.append(dp)
            pass
        pass
        return xdp

    def s5_titles(self): 
        lines = []
        for i in range(len(self.s5ti)):
            s5t = self.s5ti[i]
            title = ":i:`%s`" % s5t 
            undertitle = "-" * (3+len(title))
            lines.append(title)
            lines.append(undertitle)
            lines.append("")
        pass
        return "\n".join(lines)

    def __repr__(self):
        lines = []
        indent = self.INDENT
        assert len(self.s5ps) == len(self.s5ti)
        for i in range(len(self.s5ps)):
            s5p = self.s5ps[i]
            s5t = self.s5ti[i]
            lines.append(indent)
            lines.append(indent+s5t)
            lines.append(indent+s5p+" 1280px_720px")
        pass        
        return "\n".join(lines)

    def __str__(self):
        return "\n".join(self.cmds)
        #return "\n".join(self.res)


def parse_args(doc, **kwa):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument( "--level", default="info", help="logging level" ) 
    parser.add_argument( "args", nargs="*", help="arg" )
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args  



if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     pargs = parse_args(__doc__)
     args = pargs.args 

     p = Pub()
     for arg in args:
         if arg == "cp":
             print(p)
         elif arg == "help":
             print(__doc__)
         elif arg == "s5":
             print(repr(p))
         elif arg == "t5":
             print(p.s5_titles())
         elif arg == "sp":
             print("\n".join(p.spaths()))
         elif arg == "dp":
             print("\n".join(p.dpaths()))
         elif arg == "xdp":
             print("\n".join(p.existing_dpaths()))
         else:
             print(p)
             print(repr(p))
         pass
     pass

