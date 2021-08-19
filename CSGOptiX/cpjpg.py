#!/usr/bin/env python

import os, sys, logging
log = logging.getLogger(__name__)

class Place(object):
    DESTBASE = os.path.expandvars("$HOME/simoncblyth.bitbucket.io/env/presentation/CSGOptiXRender")
    DNAME = os.path.basename(DESTBASE)
    def __init__(self, tmp_jpgpath):
        """
        :param tmp_jpgpath:
        """
        src = tmp_jpgpath
        elem = src.split("/")
        div = elem.index(self.DNAME)
        rsrc = "/".join(elem[div+1:])
        dst = "/".join([self.DESTBASE, rsrc]) 

        log.debug("src  : %s "%  src )
        log.debug("rsrc : %s " % rsrc )
        log.debug("dst  : %s "%  dst )

        dstdir= os.path.dirname(dst) 

        cmds = []
        cmds.append("mkdir -p %s " % os.path.dirname(dst))
        cmds.append("cp %s %s " % (src, dst) )
        cmds.append("cp %s %s " % (src.replace(".jpg",".json"), dst.replace(".jpg",".json")) )
        self.cmds = cmds 

    def __str__(self):
        return "\n".join(self.cmds) 


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     default_tmp_jpgpath = "/tmp/blyth/opticks/CSGOptiX/CSGOptiXRender/CSG_GGeo/cvd1/70000/cxr_overview/cam_0_tmin_0.4/cxr_overview_emm_t8,_moi_-1.jpg"
     tmp_jpgpath = sys.argv[1] if len(sys.argv) > 1 else default_tmp_jpgpath  
     pl = Place(tmp_jpgpath)
     print(pl)



