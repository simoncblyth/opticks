multiple sensitive materials
===================================

Updating to current JUNO geometry runs into cannot change cathode issue.
Testing in integrated build environment with **tds**

::

    [blyth@localhost opticks]$ t jre
    jre is a function
    jre () 
    { 
        : setup the runtime environment CMAKE_PREFIX_PATH, PKG_CONFIG_PATH, LD_LIBRARY_PATH;
        echo jre [;
        source $JUNOTOP/bashrc.sh;
        source $JUNOTOP/sniper/SniperRelease/cmt/setup.sh;
        source $JUNOTOP/offline/JunoRelease/cmt/setup.sh;
        echo jre.gob [;
        source $JUNOTOP/offline/Simulation/DetSimV2/G4OpticksBridge/cmt/setup.sh;
        echo jre.gob ];
        function pyj () 
        { 
            python $JUNOTOP/offline/Simulation/DetSimV2/G4OpticksBridge/share/pyjob_acrylic.py
        };
        function tds () 
        { 
            export X4PhysicalVolume=INFO;
            export GMaterialLib=INFO;
            export GGeoSensor=INFO;
            python $JUNOTOP/offline/Examples/Tutorial/share/tut_detsim.py --opticks gun $*
        };
        echo jre ]
    }




::

    2020-06-02 17:35:01.439 INFO  [403208] [Opticks::init@404] COMPUTE_MODE compute_requested  forced_compute  hostname localhost.localdomain
    2020-06-02 17:35:01.439 INFO  [403208] [Opticks::init@413]  non-legacy mode : ie mandatory keyed access to geometry, opticksaux 
    2020-06-02 17:35:01.445 INFO  [403208] [BOpticksResource::setupViaKey@828] 
                 BOpticksKey  : KEYSOURCE
          spec (OPTICKS_KEY)  : G4OpticksAnaMgr.X4PhysicalVolume.pWorld.9039109b5e8f9822b5c07174678e1662
                     exename  : G4OpticksAnaMgr
             current_exename  : G4OpticksAnaMgr
                       class  : X4PhysicalVolume
                     volname  : pWorld
                      digest  : 9039109b5e8f9822b5c07174678e1662
                      idname  : G4OpticksAnaMgr_pWorld_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-06-02 17:35:01.446 INFO  [403208] [Opticks::loadOriginCacheMeta@1819]  cachemetapath /home/blyth/.opticks/geocache/G4OpticksAnaMgr_pWorld_g4live/g4ok_gltf/9039109b5e8f9822b5c07174678e1662/1/cachemeta.json
    2020-06-02 17:35:01.446 INFO  [403208] [NMeta::dump@199] Opticks::loadOriginCacheMeta
    2020-06-02 17:35:01.446 INFO  [403208] [Opticks::loadOriginCacheMeta@1823]  gdmlpath 
    2020-06-02 17:35:01.446 INFO  [403208] [G4Opticks::translateGeometry@234] ) Opticks /home/blyth/.opticks/geocache/G4OpticksAnaMgr_pWorld_g4live/g4ok_gltf/9039109b5e8f9822b5c07174678e1662/1
    2020-06-02 17:35:01.446 INFO  [403208] [G4Opticks::translateGeometry@245] ( GGeo instanciate
    2020-06-02 17:35:01.448 INFO  [403208] [G4Opticks::translateGeometry@248] ) GGeo instanciate 
    2020-06-02 17:35:01.448 INFO  [403208] [G4Opticks::translateGeometry@250] ( GGeo populate
    2020-06-02 17:35:01.448 ERROR [403208] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Galactic
    2020-06-02 17:35:01.449 ERROR [403208] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt DummyAcrylic
    2020-06-02 17:35:01.449 ERROR [403208] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Teflon
    2020-06-02 17:35:01.449 ERROR [403208] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Copper
    2020-06-02 17:35:01.450 ERROR [403208] [GMaterialLib::add@306]  MATERIAL WITH EFFICIENCY 
    2020-06-02 17:35:01.450 ERROR [403208] [GMaterialLib::add@306]  MATERIAL WITH EFFICIENCY 
    2020-06-02 17:35:01.450 FATAL [403208] [GMaterialLib::setCathode@1167]  not expecting to change cathode GMaterial from  photocathode to photocathode_3inch
    python: /home/blyth/opticks/ggeo/GMaterialLib.cc:1172: void GMaterialLib::setCathode(GMaterial*): Assertion `0' failed.
    Aborted (core dumped)
    [blyth@localhost ~]$ 

::

    [blyth@localhost opticks]$ o
    # On branch master
    # Changes not staged for commit:
    #   (use "git add <file>..." to update what will be committed)
    #   (use "git checkout -- <file>..." to discard changes in working directory)
    #
    #   modified:   assimprap/AssimpGGeo.cc
    #   modified:   cfg4/CDetector.cc
    #   modified:   extg4/X4PhysicalVolume.cc
    #   modified:   ggeo/GGeo.cc
    #   modified:   ggeo/GGeo.hh
    #   modified:   ggeo/GGeoSensor.cc
    #   modified:   ggeo/GMaterialLib.cc
    #   modified:   ggeo/GMaterialLib.hh
    #
    # Untracked files:
    #   (use "git add <file>..." to include in what will be committed)
    #
    #   .gitignore
    #   notes/issues/multiple-sensitive-materials.rst
        no changes added to commit (use "git add" and/or "git commit -a")




Adding the LVSDMT handling and removing OLD_CATHODE gets further::

    2020-06-02 22:17:10.874 INFO  [411548] [X4PhysicalVolume::init@181]  query :  queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2020-06-02 22:17:10.874 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Galactic
    2020-06-02 22:17:10.875 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt DummyAcrylic
    2020-06-02 22:17:10.875 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Teflon
    2020-06-02 22:17:10.876 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Copper
    2020-06-02 22:17:10.876 ERROR [411548] [GMaterialLib::add@310]  MATERIAL WITH EFFICIENCY 
    2020-06-02 22:17:10.876 INFO  [411548] [GMaterialLib::addSensitiveMaterial@1184]  add sensitive material  GMaterial : 0x33708570 name : photocathode
    2020-06-02 22:17:10.876 ERROR [411548] [GMaterialLib::add@310]  MATERIAL WITH EFFICIENCY 
    2020-06-02 22:17:10.876 INFO  [411548] [GMaterialLib::addSensitiveMaterial@1184]  add sensitive material  GMaterial : 0x3370c280 name : photocathode_3inch
    2020-06-02 22:17:10.877 ERROR [411548] [GMaterialLib::add@310]  MATERIAL WITH EFFICIENCY 
    2020-06-02 22:17:10.877 INFO  [411548] [GMaterialLib::addSensitiveMaterial@1184]  add sensitive material  GMaterial : 0x3370ff40 name : photocathode_MCP20inch
    2020-06-02 22:17:10.877 ERROR [411548] [GMaterialLib::add@310]  MATERIAL WITH EFFICIENCY 
    2020-06-02 22:17:10.877 INFO  [411548] [GMaterialLib::addSensitiveMaterial@1184]  add sensitive material  GMaterial : 0x33713bc0 name : photocathode_MCP8inch
    2020-06-02 22:17:10.877 ERROR [411548] [GMaterialLib::add@310]  MATERIAL WITH EFFICIENCY 
    2020-06-02 22:17:10.877 INFO  [411548] [GMaterialLib::addSensitiveMaterial@1184]  add sensitive material  GMaterial : 0x33717900 name : photocathode_Ham20inch
    2020-06-02 22:17:10.877 ERROR [411548] [GMaterialLib::add@310]  MATERIAL WITH EFFICIENCY 
    2020-06-02 22:17:10.877 INFO  [411548] [GMaterialLib::addSensitiveMaterial@1184]  add sensitive material  GMaterial : 0x3371b5c0 name : photocathode_Ham8inch
    2020-06-02 22:17:10.877 ERROR [411548] [GMaterialLib::add@310]  MATERIAL WITH EFFICIENCY 
    2020-06-02 22:17:10.877 INFO  [411548] [GMaterialLib::addSensitiveMaterial@1184]  add sensitive material  GMaterial : 0x3371f270 name : photocathode_HZC9inch
    2020-06-02 22:17:10.877 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt SiO2
    2020-06-02 22:17:10.877 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt B2O2
    2020-06-02 22:17:10.877 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Na2O
    2020-06-02 22:17:10.878 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Scintillator
    2020-06-02 22:17:10.878 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Adhesive
    2020-06-02 22:17:10.878 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Aluminium
    2020-06-02 22:17:10.878 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt TiO2
    2020-06-02 22:17:10.878 ERROR [411548] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt TiO2Coating
    2020-06-02 22:17:10.878 INFO  [411548] [GMaterialLib::beforeClose@716] .
    2020-06-02 22:17:10.878 INFO  [411548] [GMaterialLib::replaceGROUPVEL@743] GMaterialLib::replaceGROUPVEL  ni 39
    2020-06-02 22:17:10.879 INFO  [411548] [GMaterialLib::sort@476] ORDER_BY_PREFERENCE
    2020-06-02 22:17:10.879 INFO  [411548] [GMaterialLib::createMeta@521] .
    python: /home/blyth/opticks/sysrap/SAbbrev.cc:63: void SAbbrev::init(): Assertion `isFree(ab) && "failed to abbreviate "' failed.
    Aborted (core dumped)



Name abbreviation issue.








::

    2020-06-02 18:53:55.897 INFO  [86487] [Opticks::loadOriginCacheMeta@1819]  cachemetapath /home/blyth/.opticks/geocache/G4OpticksAnaMgr_pWorld_g4live/g4ok_gltf/9039109b5e8f9822b5c07174678e1662/1/cachemeta.json
    2020-06-02 18:53:55.897 INFO  [86487] [NMeta::dump@199] Opticks::loadOriginCacheMeta
    2020-06-02 18:53:55.897 INFO  [86487] [Opticks::loadOriginCacheMeta@1823]  gdmlpath 
    2020-06-02 18:53:55.898 INFO  [86487] [G4Opticks::translateGeometry@234] ) Opticks /home/blyth/.opticks/geocache/G4OpticksAnaMgr_pWorld_g4live/g4ok_gltf/9039109b5e8f9822b5c07174678e1662/1
    2020-06-02 18:53:55.898 INFO  [86487] [G4Opticks::translateGeometry@245] ( GGeo instanciate
    2020-06-02 18:53:55.899 INFO  [86487] [G4Opticks::translateGeometry@248] ) GGeo instanciate 
    2020-06-02 18:53:55.900 INFO  [86487] [G4Opticks::translateGeometry@250] ( GGeo populate
    2020-06-02 18:53:55.900 INFO  [86487] [X4PhysicalVolume::init@180] [
    2020-06-02 18:53:55.900 INFO  [86487] [X4PhysicalVolume::init@181]  query :  queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2020-06-02 18:53:55.900 ERROR [86487] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Galactic
    2020-06-02 18:53:55.901 ERROR [86487] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt DummyAcrylic
    2020-06-02 18:53:55.901 ERROR [86487] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Teflon
    2020-06-02 18:53:55.901 ERROR [86487] [X4MaterialTable::init@87] PROCEEDING TO convert material with no mpt Copper
    2020-06-02 18:53:55.902 ERROR [86487] [GMaterialLib::add@306]  MATERIAL WITH EFFICIENCY 
    2020-06-02 18:53:55.902 ERROR [86487] [GMaterialLib::add@306]  MATERIAL WITH EFFICIENCY 
    2020-06-02 18:53:55.902 FATAL [86487] [GMaterialLib::setCathode@1167]  not expecting to change cathode GMaterial from  photocathode to photocathode_3inch
    python: /home/blyth/opticks/ggeo/GMaterialLib.cc:1172: void GMaterialLib::setCathode(GMaterial*): Assertion `0' failed.
    Aborted (core dumped)
    [blyth@localhost ~]$ 








::

     300 // invoked pre-cache by GGeo::add(GMaterial* material) AssimpGGeo::convertMaterials
     301 void GMaterialLib::add(GMaterial* mat)
     302 {
     303     if(mat->hasProperty("EFFICIENCY"))
     304     {
     305         //LOG(LEVEL) << " MATERIAL WITH EFFICIENCY " ; 
     306         LOG(error) << " MATERIAL WITH EFFICIENCY " ;
     307         setCathode(mat) ;
     308     }
     309 
     310     bool with_lowercase_efficiency = mat->hasProperty("efficiency") ;
     311     assert( !with_lowercase_efficiency );
     312 
     313     assert(!isClosed());
     314     m_materials.push_back(createStandardMaterial(mat));
     315 }


* vague recollection : opticks model has sensitive surfaces, not materials : so there is some translation happening 


::

    1157 void GMaterialLib::setCathode(GMaterial* cathode)
    1158 {
    1159     assert( cathode ) ;
    1160     if( cathode && m_cathode && cathode == m_cathode )
    1161     {
    1162         LOG(fatal) << " have already set that cathode GMaterial : " << cathode->getName() ;
    1163         return ;
    1164     }
    1165     if( cathode && m_cathode && cathode != m_cathode )
    1166     {
    1167         LOG(fatal) << " not expecting to change cathode GMaterial from  "
    1168                    << m_cathode->getName()
    1169                    << " to "
    1170                    << cathode->getName()
    1171                    ;
    1172         assert(0);
    1173     }
    1174     LOG(LEVEL)
    1175            << " setting cathode "
    1176            << " GMaterial : " << cathode
    1177            << " name : " << cathode->getName() ;
    1178     //cathode->Summary();       
    1179     LOG(LEVEL) << cathode->prop_desc() ;
    1180 
    1181     assert( cathode->hasNonZeroProperty("EFFICIENCY") );
    1182 
    1183     m_cathode = cathode ;
    1184     m_cathode_material_name = strdup( cathode->getName() ) ;
    1185 }
    1186 GMaterial* GMaterialLib::getCathode() const
    1187 {
    1188     return m_cathode ;
    1189 }
    1190 
    1191 const char* GMaterialLib::getCathodeMaterialName() const
    1192 {
    1193     return m_cathode_material_name ;
    1194 }



    [blyth@localhost ggeo]$ opticks-f getCathode
    ./assimprap/AssimpGGeo.cc:    if(!gg->getCathode() )
    ./assimprap/AssimpGGeo.cc:    gg->getCathode()->Summary();
    ./assimprap/AssimpGGeo.cc:        const char* sslv = gg->getCathodeLV(i);
    ./assimprap/AssimpGGeo.cc:    GMaterial* cathode = gg->getCathode() ; 
    ./assimprap/AssimpGGeo.cc:    const char* cathode_material_name = gg->getCathodeMaterialName() ;
    ./cfg4/CDetector.cc:        //const char* lvn = m_ggeo->getCathodeLV(i); 

    ./extg4/X4PhysicalVolume.cc:GMaterialLib::setCathode getCathode


    ./ggeo/GGeo.cc:GMaterial* GGeo::getCathode() const 
    ./ggeo/GGeo.cc:    return m_materiallib->getCathode() ; 
    ./ggeo/GGeo.cc:const char* GGeo::getCathodeMaterialName() const
    ./ggeo/GGeo.cc:    return m_materiallib->getCathodeMaterialName() ; 
    ./ggeo/GGeo.cc:        const char* clv2 = getCathodeLV(index); 
    ./ggeo/GGeo.cc:const char* GGeo::getCathodeLV(unsigned int index) const 
    ./ggeo/GGeo.cc:void GGeo::getCathodeLV( std::vector<std::string>& lvnames ) const 
    ./ggeo/GGeo.cc:GMaterial* GGeo::getCathodeMaterial(unsigned int index)
    ./ggeo/GGeo.hh:        GMaterial* getCathodeMaterial(unsigned int index);
    ./ggeo/GGeo.hh:        GMaterial* getCathode() const ;  
    ./ggeo/GGeo.hh:        const char* getCathodeMaterialName() const ;
    ./ggeo/GGeo.hh:        const char* getCathodeLV(unsigned int index) const ; 
    ./ggeo/GGeo.hh:        void getCathodeLV( std::vector<std::string>& lvnames ) const ;
    ./ggeo/GGeoSensor.cc:from GGeo::getNumCathodeLV GGeo::getCathodeLV.   
    ./ggeo/GGeoSensor.cc:    GMaterial* cathode_props = gg->getCathode() ; 
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
    ./ggeo/GGeoSensor.cc:        const char* sslv = gg->getCathodeLV(i);
    ./ggeo/GMaterialLib.cc:GMaterial* GMaterialLib::getCathode() const 
    ./ggeo/GMaterialLib.cc:const char* GMaterialLib::getCathodeMaterialName() const
    ./ggeo/GMaterialLib.hh:        GMaterial* getCathode() const ;  
    ./ggeo/GMaterialLib.hh:        const char* getCathodeMaterialName() const ;
    ./optickscore/Opticks.hh:       const char*          getCathode() const ;
    ./optickscore/OpticksCfg.hh:     const std::string& getCathode();
    ./optickscore/OpticksCfg.cc:const std::string& OpticksCfg<Listener>::getCathode()
    ./optickscore/Opticks.cc:const char* Opticks::getCathode() const 
    ./optickscore/Opticks.cc:    const std::string& s = m_cfg->getCathode();
    [blyth@localhost opticks]$ 

::

     324 /**
     325 GGeo::addLVSD
     326 -------------------
     327 
     328 From  
     329 
     330 1. AssimpGGeo::convertSensorsVisit
     331 2. X4PhysicalVolume::convertSensors_r
     332 
     333 
     334 Issues/TODO
     335 ~~~~~~~~~~~~~~
     336 
     337 * integrate sensor setup with the material properties, 
     338   see GMaterialLib::setCathode, GGeoSensor::AddSensorSurfaces
     339 
     340 
     341 **/
     342 
     343 void GGeo::addLVSD(const char* lv, const char* sd)
     344 {
     345    assert( lv ) ; 
     346    m_cathode_lv.insert(lv);
     347 
     348    if(sd)
     349    {
     350        if(m_lv2sd == NULL ) m_lv2sd = new NMeta ;
     351        m_lv2sd->set<std::string>(lv, sd) ;
     352    }
     353 }
     354 unsigned GGeo::getNumLVSD() const
     355 {
     356    return m_lv2sd ? m_lv2sd->getNumKeys() : 0 ;
     357 }
     358 std::pair<std::string,std::string> GGeo::getLVSD(unsigned idx) const
     359 {
     360     const char* lv = m_lv2sd->getKey(idx) ;
     361     std::string sd = m_lv2sd->get<std::string>(lv);
     362     return std::pair<std::string,std::string>( lv, sd );
     363 }
     364 



::

    [blyth@localhost extg4]$ opticks-f getLVSD
    ./cfg4/CDetector.cc:        std::pair<std::string,std::string> lvsd = m_ggeo->getLVSD(i) ; 
    ./extg4/X4PhysicalVolume.cc:    m_lvsdname(m_ok->getLVSDName()),
    ./ggeo/GGeo.cc:std::pair<std::string,std::string> GGeo::getLVSD(unsigned idx) const
    ./ggeo/GGeo.hh:        std::pair<std::string,std::string> getLVSD(unsigned idx) const ;
    ./optickscore/Opticks.hh:       const char*          getLVSDName() const ;
    ./optickscore/OpticksCfg.hh:     const std::string& getLVSDName();
    ./optickscore/OpticksCfg.cc:const std::string& OpticksCfg<Listener>::getLVSDName()
    ./optickscore/Opticks.cc:const char* Opticks::getLVSDName() const 
    ./optickscore/Opticks.cc:    const std::string& s = m_cfg->getLVSDName();
    [blyth@localhost opticks]$ 


