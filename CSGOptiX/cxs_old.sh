        




elif [ "$(uname)" == "Darwin" ]; then

    if [ "$arg" == "lrun" ] ; then 

        cd $LOGDIR 
        source ${bin}_OUTPUT_DIR.sh || exit 1  
        echo $msg lrun mode : using the output directory discerned from the last grab
        echo $msg ${bin}_OUTPUT_DIR $${bin}_OUTPUT_DIR


        opticks_key_remote_dir=$(opticks-key-remote-dir) # eg .opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1
        cvd_ver=cvd0/70000
        if [ -n "$cfbase" ]; then 
            cxsdir=$cfbase/$bin/$cvd_ver
        else
            cxsdir=$HOME/$opticks_key_remote_dir/CSG_GGeo/$bin/$cvd_ver
        fi
        if [ ! -d "$cxsdir" ]; then 
            echo $msg ERROR no cxsdir : $cxsdir  
            exit 1 
        fi
        geomdir=$cxsdir/$GEOM
        if [ ! -d "$geomdir" ]; then 
            echo $msg ERROR no geomdir : $geomdir  
            exit 1 
        fi
        export ${bin}_OUTPUT_DIR=$geomdir
        echo $msg non-lrun mode : using the output directory defined by script variable GEOM
        echo $msg ${bin}_OUTPUT_DIR $${bin}_OUTPUT_DIR



echo LOGDIR : $LOGDIR
if [ -n "$cfbase" ]; then 
   echo $msg cfbase $cfbase is defined : are using non-standard geometry 
   echo $msg cross section intersects onto non-standard geometry are stored into tmp directories
   echo $msg to grab these to laptop use cx tmp_grab.sh 
fi 



