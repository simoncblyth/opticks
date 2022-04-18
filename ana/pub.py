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
    export SRC_BASE=$TMP/CSGOptiX/CSGOptiXSimtraceTest
    pub.py $* 

Leading to destination base folder ~/simoncblyth.bitbucket.io/env/presentation/CSGOptiXSimtraceTest/

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

    @classmethod
    def FindDigest(cls, path):
        hexchar = "0123456789abcdef" 
        digest = None
        for elem in path.split("/"):
            if len(elem) == 32 and set(elem).issubset(hexchar):
                digest = elem
            pass
        pass
        return digest 


    def __init__(self, base=None, exts=[".png", ".jpg"], digestprefix=False):
        if base is None:
            base = os.environ.get("SRC_BASE", "$TMP/NonExisting")
        pass
        expect = os.path.isdir(base)
        if not expect:
            log.fatal("base directory %s does not exist, set envvar SRC_BASE to an existing directory" % base )
        pass
        assert expect
        name = os.path.basename(base)
        digest = self.FindDigest(base)
        log.debug("digest %s " % digest )

        if digestprefix == False:
            elem = [self.DEST_BASE, name]
        else:
            elem = [self.DEST_BASE, digest, name]
        pass
        dest = os.path.join(*list(filter(None,elem)))

        log.debug("elem %s " % str(elem))  # 
        log.debug("base %s " % base)  # 
        log.debug("dest %s " % dest)

        geom = os.environ.get("GEOM", "")
        self.find(base, exts, with_elem=geom)

        cmds, s5ps, s5ti = self.copy_cmds(dest)

        self.dest = dest
        self.cmds = cmds
        self.s5ps = s5ps
        self.s5ti = s5ti

    def find(self, base, exts, with_elem=""):
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
                        relpath = path[len(base)+1:] 
                        
                        if len(with_elem) > 0: 
                            elem = relpath.split("/")
                            contains_elem = with_elem in elem
                        else:
                            contains_elem = True 
                        pass
                        log.debug("relpath %s contains_elem %s with_elem %s " % (relpath, contains_elem, with_elem) ) 

                        if contains_elem: 
                            res.append(relpath)
                        pass
                    pass
                pass
            pass
        pass
        self.base = base
        self.res = sorted(res)
        log.debug("found %d res " % len(self.res)) 


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

            if len(elem) > 2:
                ## old layout 
                title = "_".join([elem[-3], elem[-2], elem[-1]])
            elif len(elem) > 1:
                title = "_".join([elem[-2], elem[-1]])
            else:
                title = elem[0]
            pass 

            log.debug("rel %s elem %d title %s " % (rel, len(elem), title))

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
    parser.add_argument( "--digestprefix", action="store_true", default=False, help="prefix with geocache digest" ) 
    parser.add_argument( "args", nargs="*", help="arg" )
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'

    loglevel = args.level.upper()
    #print("logging --level setting to %s " % loglevel ) 

    logging.basicConfig(level=getattr(logging,loglevel), format=fmt)
    return args  



if __name__ == '__main__':

     pargs = parse_args(__doc__)
     args = pargs.args 

     p = Pub(digestprefix=pargs.digestprefix)
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

