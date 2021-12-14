#!/bin/bash -l

usage(){ cat << EOU
run.sh : Loads current GGeo identified by OPTICKS_KEY, converts into CSGFoundry and saves  
===========================================================================================

::

    ./run.sh

    DUMP_RIDX=2 ./run.sh 

    ONE_PRIM_SOLID=1 ./run.sh 

    GDB=lldb_ ./run.sh -- 


Now that -e option works at CSGOptiX level where the
CF geometry is uploaded to GPU there 
is no point to do that here and bake it into the 
persisted CSGFoundry geometry. As would then need
to manage lots of different directories of CF geometry.

--gparts_transform_offset
--------------------------

Hmm without "--gparts_transform_offset" get messed up geometry 
(JUNO chimney in middle of CD) ... because of this it is included 
on the commandline below.

This reconciles a difference in geometry handling. 
In the old model there are separate transform buffers 
for each GMergedMesh : in the new CSGFoundry approach 
there is a single transform buffer for all geometry.

As a result analytic transform referencing must be 
different for the two models.

::

    epsilon:CSG_GGeo blyth$ opticks-f gparts_transform_offset 
    ./ggeo/GParts.cc:    if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    ./ggeo/GParts.cc:        LOG(LEVEL) << " --gparts_transform_offset " ; 
    ./ggeo/GParts.cc:        LOG(LEVEL) << " NOT --gparts_transform_offset " ; 
    ./optickscore/Opticks.hh:       bool isGPartsTransformOffset() const ; // --gparts_transform_offset
    ./optickscore/OpticksCfg.cc:       ("gparts_transform_offset",  "see GParts::add") ;
    ./optickscore/Opticks.cc:    return m_cfg->hasOpt("gparts_transform_offset") ;  

ggeo/GParts.cc::

    1260 /**
    1261 GParts::add
    1262 -------------
    1263 
    1264 Basis for combination of analytic geometry.
    1265 
    1266 Notice the --gparts_transform_offset option which 
    1267 is necessary for CSG_GGeo creation of CSGFoundry as in that case the 
    1268 entire geometry is treated together. 
    1269 Without it get JUNO Chimney in middle of CD !
    1270 
    1271 Whereas for pre-7 running each GMergedMesh transforms 
    1272 are handled separately, hence --gparts_transform_offset
    1273 should not be used. 
    1274 
    1275 **/
    1276 
    1277 void GParts::add(GParts* other)
    1278 {   
    ....
    1297     if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    1298     {
    1299         LOG(LEVEL) << " --gparts_transform_offset " ;
    1300         bool preserve_zero = true ;
    1301         bool preserve_signbit = true ;
    1302         other_part_buffer->addOffset(GTRANSFORM_J, GTRANSFORM_K, tranOffset, preserve_zero, preserve_signbit );
    1303         // hmm offsetting of planes needs to be done only for parts of type CSG_CONVEXPOLYHEDRON 
    1304     }
    1305     else
    1306     {
    1307         LOG(LEVEL) << " NOT --gparts_transform_offset " ;
    1308     }

EOU
}

msg="=== $BASH_SOURCE :"
bin=CSG_GGeoTest

#export ONE_PRIM_SOLID=1 # adds extra debugging solids that reuse existing prim one-by-one
export DUMP_RIDX=${DUMP_RIDX:-8} 



GDBDIV=""
[ -n "$GDB" ] && GDBDIV="--"

which $bin

$GDB $bin $GDBDIV --gparts_transform_offset $*








logdir_notes(){ cat << EON

HMM: moved to getting the CFBASE at C++ level 
making the below logdir innapropriate 

Despite adding SPath::chdir it is problematic 
to use as the logfile setup is done very early within
OPTICKS_LOG ... so need a way to define the logs directory 
within the CSG_GGeo dir of the idpath exceedingly early.

Actually a logs directory within the idpath would be OK also.
So can use OPTICKS_KEY which is a known to be existing envvar so can 
create a static function that rustles up the idpath just from 
that : without all the other processing. 

HMM: could avoid needing logdir and outdir in this script
by listing from the C++ see sysrap/tests/dirent.cc 

EON
}

opticks_geocache_prefix=~/.opticks
geocache_sh=${OPTICKS_GEOCACHE_PREFIX:-$opticks_geocache_prefix}/geocache/geocache.sh  

if [ -f "$geocache_sh" ]; then 
    echo $msg geocache_sh $geocache_sh sourcing
    cat $geocache_sh
    source $geocache_sh

    cfbase=${OPTICKS_KEYDIR}/CSG_GGeo
    logdir=$cfbase/logs  # matches the chdir done in tests/CSG_GGeoTest.cc
    outdir=$cfbase/CSGFoundry

    echo $msg outdir:$outdir
    ls -l $outdir/

    echo $msg logdir:$logdir
    ls -l $logdir/

else
    echo $msg geocache_sh $geocache_sh does not exist 
    exit 0
fi 

exit 0 


