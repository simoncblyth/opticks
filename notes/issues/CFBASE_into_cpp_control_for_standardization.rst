CFBASE_into_cpp_control_for_standardization
=============================================


::

    epsilon:opticks blyth$ opticks-f CFBASE
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );  // $CFBASE/CSGFoundry must exist 
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:    // new layout : save outputs within $CFBASE/CSGOptiXSimulateTest 
    ./CSGOptiX/tests/CSGOptiXRender.cc:CFBASE
    ./CSGOptiX/tests/CSGOptiXRender.cc:    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );
    ./CSGOptiX/cxs.sh:export CFBASE=${CFBASE:-$cfbase}   ## CRITICAL CONTROL OF THE GEOMETRY TO LOAD  
    ./CSGOptiX/cxr.sh:TODO: have moved CFBASE inside idpath ... can that be handled at C++ level ?
    ./CSGOptiX/cxr.sh:export CFBASE=/tmp/$USER/opticks/${CFNAME} 
    ./CSGOptiX/cxr.sh:[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1
    ./CSGOptiX/cxr.sh:    local mname=$CFBASE/CSGFoundry/name.txt  # /tmp/$USER/opticks/CSG_GGeo/CSGFoundry/name.txt  # mesh names


    ./CSG/CSGDemoTest.sh:export CFBASE=/tmp/$USER/opticks/CSGDemoTest/$GEOMETRY
    ./CSG/CSGDemoTest.sh:cfdir=$CFBASE/CSGFoundry
    ./CSG/CSGDemoTest.sh:vars="bin GEOMETRY CLUSTERSPEC CLUSTERUNIT GRIDMODULO GRIDSINGLE GRIDSPEC GRIDSCALE LAYERS CFBASE cfdir"
    ./CSG/tests/CSGFoundryLoadTest.cc:    CSGFoundry* fd = CSGFoundry::Load(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), "CSGFoundry"); 
    ./CSG/tests/CSGPrimSpecTest.cc:    CSGFoundry* fd = CSGFoundry::Load(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), "CSGFoundry"); 
    ./CSG/tests/CSGTargetTest.cc:    CSGFoundry* fd = CSGFoundry::Load(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), "CSGFoundry"); 
    ./CSG/tests/CSGPrimTest.cc:    CSGFoundry* fd = CSGFoundry::Load(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), "CSGFoundry"); 
    ./CSG/tests/CSGDemoTest.cc:    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSGDemoTest/default" );
    ./CSG/tests/CSGDemoTest.cc:    fd.write(cfbase, rel );    // expects existing directory $CFBASE/CSGFoundry 
    ./CSG/tests/CSGNameTest.cc:    CSGFoundry* fd = CSGFoundry::Load(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), "CSGFoundry"); 



    ./bin/flight7.sh:export CFBASE=/tmp/$USER/opticks/CSG_GGeo 
    ./bin/flight7.sh:[ ! -d "$CFBASE/CSGFoundry" ] && echo $msg ERROR no such directory $CFBASE/CSGFoundry && exit 1


    ./GeoChain/GeoChain.cc:    const char* cfbase = SSys::getenvvar("CFBASE", fold  );
    ./GeoChain/GeoChain.cc:    fd->write(cfbase, rel );    // expects existing directory $CFBASE/CSGFoundry 

    ./qudarap/tests/QScintTest.cc:    const char* cfbase = SPath::Resolve(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), create_dirs );
    ./qudarap/tests/QSimTest.cc:    const char* cfbase =  SPath::Resolve(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), create_dirs ) ; 
    ./qudarap/tests/QSimWithEventTest.cc:    const char* cfbase =  SPath::Resolve(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), 0) ; 
    ./qudarap/tests/QBndTest.cc:    const char* cfbase = SPath::Resolve(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo"), create_dirs ); // 

    ./optickscore/Opticks.hh:       const char* getFoundryBase(const char* ekey="CFBASE") const ; 
    ./optickscore/Opticks.cc:Formerly controlled the location of CSGFoundry via the CFBASE envvar set in scripts such as::
    ./optickscore/Opticks.cc:Are now migrating to CFBASE default being defined in C++ to a location within the idpath.

    ./boostrap/BOpticksResource.cc:std::string BOpticksResource::getCSG_GGeoDir() const // aka CFBASE

    ./CSG_GGeo/tests/CSG_GGeoTest.cc:    const char* cfbase = ok.getFoundryBase("CFBASE"); 
    ./CSG_GGeo/tests/CSG_GGeoTest.cc:    LOG(error) << "[ write foundry to CFBASE " << cfbase << " rel " << rel  ; 
    ./CSG_GGeo/run.sh:HMM: moved to getting the CFBASE at C++ level 

    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 

