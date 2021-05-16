#!/usr/bin/env python

import sys, glob

if __name__ == '__main__':

     default_ptn = "/tmp/blyth/opticks/CSGOptiX/CSGOptiXRender/CSG_GGeo/cvd1/70000/cxr_overview/cam_0_emm_*/cxr_overview__-1.jpg"
     ptn = sys.argv[1] if len(sys.argv) > 1 else default_ptn

     paths = glob.glob(ptn)

     print("ptn : \"%s\" " % ptn )
     for path in paths:
         js = path.replace(".jpg",".json") 
         print(path)
     pass 
     

