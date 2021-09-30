#!/usr/bin/env python

import os, sys, logging
log = logging.getLogger(__name__) 

class Pub(object):
    DEST_ROOT = os.path.expanduser("~/simoncblyth.bitbucket.io")
    DEST_BASE = os.path.join(DEST_ROOT, "env/presentation")

    def __init__(self, base=None, ext=".png"):
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

        self.find(base, ext)

        cmds, s5ps = self.copy_cmds(dest)
        self.cmds = cmds
        self.s5ps = s5ps

    def find(self, base, ext):
        base = os.path.expandvars(base)
        res = []
        for root, dirs, files in os.walk(base):
            for name in files:
                if name.endswith(ext): 
                    path = os.path.join(root, name)
                    res.append(path[len(base)+1:])
                pass
            pass
        pass
        self.base = base
        self.res = sorted(res)

    def copy_cmds(self, dest):
        cmds = []
        s5ps = []
        for rel in self.res:
            spath = os.path.join(self.base, rel)
            dpath = os.path.join(dest, rel)
            ddir = os.path.dirname( dpath)
            cmds.append("mkdir -p %s " % ddir)
            cmds.append("cp %s %s " % (spath, dpath))
            s5p = "/"+dpath[len(self.DEST_ROOT)+1:]  
            s5ps.append(s5p)
        pass
        return cmds, s5ps

    def __repr__(self):
        lines = []
        indent = "    "
        for s5p in self.s5ps:
            lines.append(indent)
            lines.append(indent+"Title")
            lines.append(indent+s5p+" 1280px_720px")
        pass        
        return "\n".join(lines)

    def __str__(self):
        return "\n".join(self.cmds)
        #return "\n".join(self.res)

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     p = Pub()
     if len(sys.argv) > 1 and sys.argv[1] == "cp":
         print(p)
     elif len(sys.argv) > 1 and sys.argv[1] == "s5":
         print(repr(p))
     else:
         print(p)
         print(repr(p))
     pass

