#!/bin/bash -l 

usage(){ cat << EOU
gs.sh 
=======

Loads gensteps and dumps info about them such as numPhotons, pdgcode, position ranges::

    gs.sh 1 2 3 
    gs.sh -1 -2 -3 
    gs.sh 1 -1 
    
    gs.sh /tmp/blyth/opticks/source/evt/g4live/natural/1/gs.npy
    gs.sh /tmp/blyth/opticks/source/evt/g4live/natural/-1/gs.npy

Gensteps can be identified by full paths or integer tags that assume a path template.

See also gsplt.py 


EOU
}

ipython -i $(which gs.py) -- $*



