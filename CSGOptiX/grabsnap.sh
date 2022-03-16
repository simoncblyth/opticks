#!/bin/bash -l 
usage(){ cat << EOU
grabsnap.sh
=============

Making table from .json sidecars grabbed from remote::

   ./grabsnap.sh           ## txt table 
   ./grabsnap.sh  --rst    ## RST table

EOU
}

executable=${EXECUTABLE:-CSGOptiXRenderTest}
opticks_key_remote_dir=$(opticks-key-remote-dir)

xdir=$opticks_key_remote_dir/CSG_GGeo/$executable/   ## trailing slash to avoid duplicating path element 
from=P:$xdir
to=$HOME/$xdir

#globptn="${to}cvd1/70000/cxr_overview/cam_0_tmin_0.4/cxr_overview*.jpg"
globptn="${to}cvd1/70000/cxr_overview/cam_0_tmin_0.4/cxr_overview*elv*.jpg"
refjpgpfx="/env/presentation/cxr/cxr_overview"


mkdir -p $to

printf "arg                     %s \n " "$arg"
printf "EXECUTABLE              %s \n " "$EXECUTABLE"
printf "LOGDIR                  %s \n " "$LOGDIR"
printf "OPTICKS_KEY_REMOTE      %s \n " "$OPTICKS_KEY_REMOTE" 
printf "opticks_key_remote_dir  %s \n " "$opticks_key_remote_dir" 
printf "\n"
printf "xdir                    %s \n" "$xdir"
printf "from                    %s \n" "$from" 
printf "to                      %s \n" "$to" 
printf "globptn                 %s \n" "$globptn"
printf "refjpgpfx               %s \n" "$refjpgpfx"
  

if [ "$arg" == "grab" ]; then 

    rsync -zarv --progress --include="*/" --include="*.jpg" --include="*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.json'`
fi 

${IPYTHON:-ipython} -i $OPTICKS_HOME/ana/snap.py --  --globptn "$globptn" --refjpgpfx "$refjpgpfx" $*


