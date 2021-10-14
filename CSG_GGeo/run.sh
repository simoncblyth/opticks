#!/bin/bash -l

usage(){ cat << EOU

::

    ./run.sh

    ONE_PRIM_SOLID=1 ./run.sh 

    GDB=lldb_ ./run.sh -- 


Now that -e option works at CSGOptiX level where the
CF geometry is uploaded to GPU there 
is no point to do that here and bake it into the 
persisted CSGFoundry geometry. As would then need
to manage lots of different directories of CF geometry.

--gparts_tranform_offset
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
sdir=$(pwd)
name=$(basename $sdir) 

#export ONE_PRIM_SOLID=1 # adds extra debugging solids that reuse existing prim one-by-one

export DUMP_RIDX=8 

export CFBASE=/tmp/$USER/opticks/CSG_GGeo
outdir=${CFBASE}/CSGFoundry
logdir=${CFBASE}/logs

mkdir -p $outdir 
mkdir -p $logdir 


GDBDIV=""
[ -n "$GDB" ] && GDBDIV="--"

cd $logdir
which $name
$GDB $name $GDBDIV --gparts_transform_offset $*

echo $msg outdir:$outdir
ls -l $outdir/

echo $msg logdir:$logdir
ls -l $logdir/

pwd

