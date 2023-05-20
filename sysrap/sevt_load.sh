#!/bin/bash -l 


#nevt=0  # 0 for default single Fold loading
nevt=3  # gt 0 for concat loading 
export NEVT=${NEVT:-$nevt} 

evt="000"
[ $NEVT -gt 0 ] && evt="%0.3d"

export GEOM=V0J008
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/ntds2
export FOLD=$BASE/ALL0/${EVT:-$evt}

echo $BASH_FOLD : FOLD $FOLD NEVT $NEVT  

${IPYTHON:-ipython} --pdb -i sevt_load.py 


