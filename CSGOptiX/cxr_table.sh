#!/bin/bash -l 

globptn="/tmp/blyth/opticks/CSGOptiX/CSGOptiXRender/CSG_GGeo/cvd1/70000/cxr_overview/cam_0_tmin_0.4/cxr_overview*.jpg"

#ipython -i $(which snap.py) --  --globptn "$globptn" --refjpgpfx /env/presentation/cxr/cxr_overview $*
snap.py  --globptn "$globptn" --refjpgpfx /env/presentation/cxr/cxr_overview $*

