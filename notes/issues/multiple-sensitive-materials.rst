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




::

    2020-06-03 02:45:43.094 INFO  [367987] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 02:45:43.094 INFO  [367987] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2020-06-03 02:45:43.094 INFO  [367987] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 02:45:43.094 INFO  [367987] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2020-06-03 02:45:43.094 INFO  [367987] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    python: /home/blyth/opticks/npy/NMeta.cpp:255: const char* NMeta::getKey(unsigned int) const: Assertion `idx < m_keys.size()' failed.
    Aborted (core dumped)
    [blyth@localhost ~]$ 





::

    (gdb) bt
    #0  0x00007ffff6cfa207 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6cfb8f8 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6cf3026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6cf30d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcc71e9f1 in NMeta::getKey (this=0x33571500, idx=0) at /home/blyth/opticks/npy/NMeta.cpp:255
    #5  0x00007fffcd51ba4d in GGeo::getLVSD (this=0x33483020, idx=0) at /home/blyth/opticks/ggeo/GGeo.cc:380
    #6  0x00007fffcd51c095 in GGeo::getSensitiveLVSDMT (this=0x33483020, lvn=std::vector of length 6, capacity 8 = {...}, sdn=std::vector of length 0, capacity 0, mtn=std::vector of length 0, capacity 0)
        at /home/blyth/opticks/ggeo/GGeo.cc:477
    #7  0x00007fffcd4f7e5e in GGeoSensor::AddSensorSurfaces (gg=0x33483020) at /home/blyth/opticks/ggeo/GGeoSensor.cc:104
    #8  0x00007fffce7dfea8 in X4PhysicalVolume::convertSensors (this=0x7fffffff3680) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:219
    #9  0x00007fffce7dfc2a in X4PhysicalVolume::init (this=0x7fffffff3680) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:186
    #10 0x00007fffce7df9c4 in X4PhysicalVolume::X4PhysicalVolume (this=0x7fffffff3680, ggeo=0x33483020, top=0x2cc9980) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:170
    #11 0x00007fffcf08b64f in G4Opticks::translateGeometry (this=0x20f2b950, top=0x2cc9980) at /home/blyth/opticks/g4ok/G4Opticks.cc:251
    #12 0x00007fffcf08a8bd in G4Opticks::setGeometry (this=0x20f2b950, world=0x2cc9980, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:168
    #13 0x00007fffcf08a377 in G4Opticks::Initialize (world=0x2cc9980, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:103
    #14 0x00007fffcf2b6b62 in G4OpticksAnaMgr::BeginOfRunAction (this=0x255a130, aRun=0x20f2b600) at ../src/G4OpticksAnaMgr.cc:42
    #15 0x00007fffc056057a in MgrOfAnaElem::BeginOfRunAction (this=0x7fffc076c3a0 <MgrOfAnaElem::instance()::s_mgr>, run=0x20f2b600) at ../src/MgrOfAnaElem.cc:33
    #16 0x00007fffc0d3bd6c in LSExpRunAction::BeginOfRunAction (this=0x2c64d10, aRun=0x20f2b600) at ../src/LSExpRunAction.cc:54
    #17 0x00007fffd006ae38 in G4RunManager::RunInitialization() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #18 0x00007fffc0f756fc in G4SvcRunManager::initializeRM() () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libG4Svc.so
    #19 0x00007fffc0559496 in DetSimAlg::initialize (this=0x2559640) at ../src/DetSimAlg.cc:78
    #20 0x00007fffefcbf228 in DleSupervisor::initialize() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so
    #21 0x00007fffefcc9c68 in Task::initialize() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so
    #22 0x00007fffefcceda6 in TaskWatchDog::initialize() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so
    #23 0x00007fffefcc9b7f in Task::run() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so
    #24 0x00007ffff00077dc in TaskWrap::default_run() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperPython.so


    (gdb) f 7
    #7  0x00007fffcd4f7e5e in GGeoSensor::AddSensorSurfaces (gg=0x33483020) at /home/blyth/opticks/ggeo/GGeoSensor.cc:104
    104     gg->getSensitiveLVSDMT(lvn, sdn, mtn); 
    (gdb) p lvn
    $1 = std::vector of length 6, capacity 8 = {"PMT_20inch_veto_inner1_log", "PMT_3inch_inner1_log", "PMT_20inch_veto_body_log", "PMT_3inch_body_log", "NNVTMCPPMT_PMT_20inch_inner1_log", 
      "NNVTMCPPMT_PMT_20inch_body_log"}
    (gdb) p sdn
    $2 = std::vector of length 0, capacity 0
    (gdb) p mtn
    $3 = std::vector of length 0, capacity 0
    (gdb) 

    (gdb) f 5
    #5  0x00007fffcd51ba4d in GGeo::getLVSD (this=0x33483020, idx=0) at /home/blyth/opticks/ggeo/GGeo.cc:380
    380     const char* lv = m_lv2sd->getKey(idx) ; 
    (gdb) p m_lv2sd
    $4 = (NMeta *) 0x33571500
    (gdb) p m_lv2sd->dumpLines(0)
    2020-06-03 03:12:28.785 INFO  [391251] [NMeta::dumpLines@136] (null)
    NNVTMCPPMT_PMT_20inch_body_log : "PMTSDMgr"
    NNVTMCPPMT_PMT_20inch_inner1_log : "PMTSDMgr"
    PMT_20inch_veto_body_log : "PMTSDMgr"
    PMT_20inch_veto_inner1_log : "PMTSDMgr"
    PMT_3inch_body_log : "PMTSDMgr"
    PMT_3inch_inner1_log : "PMTSDMgr"
    $5 = void
    (gdb) 


    (gdb) f 5
    #5  0x00007fffcd51ba4d in GGeo::getLVSD (this=0x33483020, idx=0) at /home/blyth/opticks/ggeo/GGeo.cc:380
    380     const char* lv = m_lv2sd->getKey(idx) ; 
    (gdb) p m_lv2sd
    $4 = (NMeta *) 0x33571500
    (gdb) p m_lv2sd->dumpLines(0)
    2020-06-03 03:12:28.785 INFO  [391251] [NMeta::dumpLines@136] (null)
    NNVTMCPPMT_PMT_20inch_body_log : "PMTSDMgr"
    NNVTMCPPMT_PMT_20inch_inner1_log : "PMTSDMgr"
    PMT_20inch_veto_body_log : "PMTSDMgr"
    PMT_20inch_veto_inner1_log : "PMTSDMgr"
    PMT_3inch_body_log : "PMTSDMgr"
    PMT_3inch_inner1_log : "PMTSDMgr"
    $5 = void
    (gdb) f 4
    #4  0x00007fffcc71e9f1 in NMeta::getKey (this=0x33571500, idx=0) at /home/blyth/opticks/npy/NMeta.cpp:255
    255     assert( idx < m_keys.size() );
    (gdb) p m_keys.size()
    $6 = 0
    (gdb) p this->dumpLines(0)
    2020-06-03 03:15:33.148 INFO  [391251] [NMeta::dumpLines@136] (null)
    NNVTMCPPMT_PMT_20inch_body_log : "PMTSDMgr"
    NNVTMCPPMT_PMT_20inch_inner1_log : "PMTSDMgr"
    PMT_20inch_veto_body_log : "PMTSDMgr"
    PMT_20inch_veto_inner1_log : "PMTSDMgr"
    PMT_3inch_body_log : "PMTSDMgr"
    PMT_3inch_inner1_log : "PMTSDMgr"
    $7 = void
    (gdb) p this->desc(10)
    $8 = "{\n          \"NNVTMCPPMT_PMT_20inch_body_log\": \"PMTSDMgr\",\n          \"NNVTMCPPMT_PMT_20inch_inner1_log\": \"PMTSDMgr\",\n          \"PMT_20inch_veto_body_log\": \"PMTSDMgr\",\n          \"PMT_20inch_veto_inner1_"...
    (gdb) p this->getNumKeys()
    $9 = 6
    (gdb) p m_keys.size()
    $10 = 6
    (gdb) p m_keys
    $11 = std::vector of length 6, capacity 8 = {"NNVTMCPPMT_PMT_20inch_body_log", "NNVTMCPPMT_PMT_20inch_inner1_log", "PMT_20inch_veto_body_log", "PMT_20inch_veto_inner1_log", "PMT_3inch_body_log", 
      "PMT_3inch_inner1_log"}
    (gdb) 




    2020-06-03 03:30:21.215 INFO  [440410] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 03:30:21.215 INFO  [440410] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2020-06-03 03:30:21.215 INFO  [440410] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 03:30:21.215 INFO  [440410] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2:020-06-03 03:30:21.215 INFO  [440410] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    python: /home/blyth/opticks/ggeo/GGeo.cc:487: void GGeo::getSensitiveLVSDMT(std::vector<std::basic_string<char> >&, std::vector<std::basic_string<char> >&, std::vector<std::basic_string<char> >&) const: Assertion `strcmp(lv, lv0) == 0' failed.
    
    Program received signal SIGABRT, Aborted.
    0x00007ffff6cfa207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXmu-1.1.2-2.el7.x86_64 libXt-1.1.5-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-57.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 mesa-libGLU-9.0.0-4.el7.x86_64 ncurses-libs-5.9-14.20130511.el7_4.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-21.el7_6.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 xz-libs-5.2.2-1.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff6cfa207 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6cfb8f8 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6cf3026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6cf30d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcd51c1b0 in GGeo::getSensitiveLVSDMT (this=0x33483020, lvn=std::vector of length 6, capacity 8 = {...}, sdn=std::vector of length 0, capacity 0, mtn=std::vector of length 0, capacity 0)
        at /home/blyth/opticks/ggeo/GGeo.cc:487
    #5  0x00007fffcd4f7e5e in GGeoSensor::AddSensorSurfaces (gg=0x33483020) at /home/blyth/opticks/ggeo/GGeoSensor.cc:104
    #6  0x00007fffce7dfea8 in X4PhysicalVolume::convertSensors (this=0x7fffffff3680) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:219
    #7  0x00007fffce7dfc2a in X4PhysicalVolume::init (this=0x7fffffff3680) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:186
    #8  0x00007fffce7df9c4 in X4PhysicalVolume::X4PhysicalVolume (this=0x7fffffff3680, ggeo=0x33483020, top=0x2cc9980) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:170
    #9  0x00007fffcf08b64f in G4Opticks::translateGeometry (this=0x20f2b950, top=0x2cc9980) at /home/blyth/opticks/g4ok/G4Opticks.cc:251
    #10 0x00007fffcf08a8bd in G4Opticks::setGeometry (this=0x20f2b950, world=0x2cc9980, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:168
    #11 0x00007fffcf08a377 in G4Opticks::Initialize (world=0x2cc9980, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:103
    #12 0x00007fffcf2b6b62 in G4OpticksAnaMgr::BeginOfRunAction (this=0x255a130, aRun=0x20f2b600) at ../src/G4OpticksAnaMgr.cc:42
    #13 0x00007fffc056057a in MgrOfAnaElem::BeginOfRunAction (this=0x7fffc076c3a0 <MgrOfAnaElem::instance()::s_mgr>, run=0x20f2b600) at ../src/MgrOfAnaElem.cc:33
    #14 0x00007fffc0d3bd6c in LSExpRunAction::BeginOfRunAction (this=0x2c64d10, aRun=0x20f2b600) at ../src/LSExpRunAction.cc:54
    #15 0x00007fffd006ae38 in G4RunManager::RunInitialization() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
v


    (gdb) f 4
    #4  0x00007fffcd51c1b0 in GGeo::getSensitiveLVSDMT (this=0x33483020, lvn=std::vector of length 6, capacity 8 = {...}, sdn=std::vector of length 0, capacity 0, mtn=std::vector of length 0, capacity 0)
        at /home/blyth/opticks/ggeo/GGeo.cc:487
    487         assert( strcmp(lv, lv0) == 0 ); 
    (gdb) p lv
    $1 = 0x3359cae8 "PMT_20inch_veto_inner1_log"
    (gdb) p lv0
    $2 = 0x3359c2a8 "NNVTMCPPMT_PMT_20inch_body_log"
    (gdb) p lv1
    $3 = 0x3359c268 "NNVTMCPPMT_PMT_20inch_body_log"
    (gdb) 



    2020-06-03 03:59:16.127 INFO  [26616] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 03:59:16.128 INFO  [26616] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2020-06-03 03:59:16.128 INFO  [26616] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 03:59:16.128 INFO  [26616] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2020-06-03 03:59:16.128 INFO  [26616] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 03:59:16.128 INFO  [26616] [GGeoSensor::AddSensorSurfaces@141]  i 0 sslv PMT_20inch_veto_inner1_log sd PMT_20inch_veto_inner1_log mt Vacuum index 71 num_mat 39 num_sks 3 num_bds 29
    2020-06-03 03:59:16.128 INFO  [26616] [GGeoSensor::MakeOpticalSurface@191]  sslv PMT_20inch_veto_inner1_log name PMT_20inch_veto_inner1_logSensorSurface
    python: /home/blyth/opticks/ggeo/GPropertyMap.cc:300: void GPropertyMap<T>::setSensor(bool) [with T = float]: Assertion `0 && "sensors are now detected by the prescense of an EFFICIENCY property"' failed.
    
    Program received signal SIGABRT, Aborted.
    0x00007ffff6cfa207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXmu-1.1.2-2.el7.x86_64 libXt-1.1.5-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-57.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 mesa-libGLU-9.0.0-4.el7.x86_64 ncurses-libs-5.9-14.20130511.el7_4.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-21.el7_6.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 xz-libs-5.2.2-1.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    i
    
    (gdb) bt
    #0  0x00007ffff6cfa207 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6cfb8f8 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6cf3026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6cf30d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcd4948c4 in GPropertyMap<float>::setSensor (this=0x3359d240, sensor=true) at /home/blyth/opticks/ggeo/GPropertyMap.cc:300
    #5  0x00007fffcd4f8334 in GGeoSensor::AddSensorSurfaces (gg=0x33483020) at /home/blyth/opticks/ggeo/GGeoSensor.cc:154
    #6  0x00007fffce7dfea8 in X4PhysicalVolume::convertSensors (this=0x7fffffff3680) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:219
    #7  0x00007fffce7dfc2a in X4PhysicalVolume::init (this=0x7fffffff3680) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:186
    #8  0x00007fffce7df9c4 in X4PhysicalVolume::X4PhysicalVolume (this=0x7fffffff3680, ggeo=0x33483020, top=0x2cc9980) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:170
    #9  0x00007fffcf08b64f in G4Opticks::translateGeometry (this=0x20f2b950, top=0x2cc9980) at /home/blyth/opticks/g4ok/G4Opticks.cc:251
    #10 0x00007fffcf08a8bd in G4Opticks::setGeometry (this=0x20f2b950, world=0x2cc9980, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:168
    #11 0x00007fffcf08a377 in G4Opticks::Initialize (world=0x2cc9980, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:103
    #12 0x00007fffcf2b6b62 in G4OpticksAnaMgr::BeginOfRunAction (this=0x255a130, aRun=0x20f2b600) at ../src/G4OpticksAnaMgr.cc:42
    #13 0x00007fffc056057a in MgrOfAnaElem::BeginOfRunAction (this=0x7fffc076c3a0 <MgrOfAnaElem::instance()::s_mgr>, run=0x20f2b600) at ../src/MgrOfAnaElem.cc:33
    #14 0x00007fffc0d3bd6c in LSExpRunAction::BeginOfRunAction (this=0x2c64d10, aRun=0x20f2b600) at ../src/LSExpRunAction.cc:54
    #15 0x00007fffd006ae38 in G4RunManager::RunInitialization() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #16 0x00007fffc0f756fc in G4SvcRunManager::initializeRM() () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libG4Svc.so
    #17 0x00007fffc0559496 in DetSimAlg::initialize (this=0x2559640) at ../src/DetSimAlg.cc:78


    2020-06-03 19:00:11.873 INFO  [30318] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2020-06-03 19:00:11.873 INFO  [30318] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 19:00:11.873 INFO  [30318] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_body_log nameref PMT_20inch_veto_body_log mt_name Pyrex
    2020-06-03 19:00:11.873 INFO  [30318] [X4PhysicalVolume::convertSensors_r@308]  is_lvsdname 0 is_sd 1 sdn PMTSDMgr name PMT_20inch_veto_inner1_log nameref PMT_20inch_veto_inner1_log mt_name Vacuum
    2020-06-03 19:00:11.874 INFO  [30318] [GGeoSensor::AddSensorSurfaces@141]  i 0 sslv PMT_20inch_veto_inner1_log sd PMT_20inch_veto_inner1_log mt Vacuum index 71 num_mat 39 num_sks 3 num_bds 29
    2020-06-03 19:00:11.874 INFO  [30318] [GGeoSensor::MakeOpticalSurface@192]  sslv PMT_20inch_veto_inner1_log name PMT_20inch_veto_inner1_logSensorSurface
    2020-06-03 19:00:11.874 INFO  [30318] [GGeoSensor::AddSensorSurfaces@162]  gss GSS:: GPropertyMap<T>:: 71    skinsurface s: GOpticalSurface  type 0 model 1 finish 3 value     1PMT_20inch_veto_inner1_logSensorSurface k:refractive_index absorption_length scattering_length reemission_prob group_velocity extra_y extra_z extra_w
    python: /home/blyth/opticks/ggeo/GSurfaceLib.cc:610: GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>*): Assertion `_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "' failed.
    
    Program received signal SIGABRT, Aborted.
    0x00007ffff6cfa207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXmu-1.1.2-2.el7.x86_64 libXt-1.1.5-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-57.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 mesa-libGLU-9.0.0-4.el7.x86_64 ncurses-libs-5.9-14.20130511.el7_4.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-21.el7_6.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 xz-libs-5.2.2-1.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff6cfa207 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6cfb8f8 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6cf3026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6cf30d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcd4b7d8f in GSurfaceLib::createStandardSurface (this=0x33488170, src=0x3359d750) at /home/blyth/opticks/ggeo/GSurfaceLib.cc:610
    #5  0x00007fffcd4b7428 in GSurfaceLib::add (this=0x33488170, surf=0x3359d750) at /home/blyth/opticks/ggeo/GSurfaceLib.cc:485
    #6  0x00007fffcd4b722b in GSurfaceLib::addSkinSurface (this=0x33488170, surf=0x3359d750, sslv_=0x3359d4a0 "PMT_20inch_veto_inner1_log", direct=false) at /home/blyth/opticks/ggeo/GSurfaceLib.cc:457
    #7  0x00007fffcd4b7112 in GSurfaceLib::add (this=0x33488170, raw=0x3359d750) at /home/blyth/opticks/ggeo/GSurfaceLib.cc:445
    #8  0x00007fffcd51b460 in GGeo::add (this=0x334851e0, surface=0x3359d750) at /home/blyth/opticks/ggeo/GGeo.cc:212
    #9  0x00007fffcd4f83fb in GGeoSensor::AddSensorSurfaces (gg=0x334851e0) at /home/blyth/opticks/ggeo/GGeoSensor.cc:164
    #10 0x00007fffce7dfea8 in X4PhysicalVolume::convertSensors (this=0x7fffffff5300) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:219
    #11 0x00007fffce7dfc2a in X4PhysicalVolume::init (this=0x7fffffff5300) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:186
    #12 0x00007fffce7df9c4 in X4PhysicalVolume::X4PhysicalVolume (this=0x7fffffff5300, ggeo=0x334851e0, top=0x2cc66b0) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:170
    #13 0x00007fffcf08b64f in G4Opticks::translateGeometry (this=0x20f27bc0, top=0x2cc66b0) at /home/blyth/opticks/g4ok/G4Opticks.cc:251
    #14 0x00007fffcf08a8bd in G4Opticks::setGeometry (this=0x20f27bc0, world=0x2cc66b0, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:168
    #15 0x00007fffcf08a377 in G4Opticks::Initialize (world=0x2cc66b0, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:103
    #16 0x00007fffcf2b6b62 in G4OpticksAnaMgr::BeginOfRunAction (this=0x2556ef0, aRun=0x20f27870) at ../src/G4OpticksAnaMgr.cc:42
    #17 0x00007fffc056057a in MgrOfAnaElem::BeginOfRunAction (this=0x7fffc076c3a0 <MgrOfAnaElem::instance()::s_mgr>, run=0x20f27870) at ../src/MgrOfAnaElem.cc:33
    #18 0x00007fffc0d3bd6c in LSExpRunAction::BeginOfRunAction (this=0x2c61a40, aRun=0x20f27870) at ../src/LSExpRunAction.cc:54
    #19 0x00007fffd006ae38 in G4RunManager::RunInitialization() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #20 0x00007fffc0f756fc in G4SvcRunManager::initializeRM() () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libG4Svc.so
    #21 0x00007fffc0559496 in DetSimAlg::initialize (this=0x25563c0) at ../src/DetSimAlg.cc:78
    #22 0x00007fffefcbf228 in DleSupervisor::initialize() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so


    (gdb) l
    480 **/
    481 
    482 void GSurfaceLib::add(GPropertyMap<float>* surf)
    483 {
    484     assert(!isClosed());
    485     GPropertyMap<float>* ssurf = createStandardSurface(surf) ;
    486     addDirect(ssurf);
    487 }
    488 
    489 void GSurfaceLib::addDirect(GPropertyMap<float>* surf)
    (gdb) f 4
    #4  0x00007fffcd4b7d8f in GSurfaceLib::createStandardSurface (this=0x33488170, src=0x3359d750) at /home/blyth/opticks/ggeo/GSurfaceLib.cc:610
    610             assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );
    (gdb) l
    605             }
    606         }
    607         else
    608         {
    609             GProperty<float>* _REFLECTIVITY = src->getProperty(REFLECTIVITY); 
    610             assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );
    611 
    612             if(os->isSpecular())
    613             {
    614                 _detect  = makeConstantProperty(0.0) ;    
    (gdb) p src
    $1 = (GPropertyMap<float> *) 0x3359d750
    (gdb) p src->desc()
    $2 = " GPropertyMap  type     skinsurface name PMT_20inch_veto_inner1_logSensorSurface isSS 1 isBS 0 isTS 0 isSU 1 isMT 0 sslv PMT_20inch_veto_inner1_log"
    (gdb) p src->prop_desc()
    $3 = " typ skinsurface idx   71 dig 605aa92893a85b895f7c827ea30410ee npr  8 nam PMT_20inch_veto_inner1_logSensorSurface\nrefractive_index :  constant: 1\nabsorption_length :  constant: 1e+09\nscattering_length"...
    (gdb) 

    (gdb) p src->m_keys
    $4 = std::vector of length 8, capacity 8 = {"refractive_index", "absorption_length", "scattering_length", "reemission_prob", "group_velocity", "extra_y", "extra_z", "extra_w"}
    (gdb) 



* mixup between origin and translated props ?
* maybe should be passing the raw material props with efficiency in ?

::

     587         if(src->isSensor())
     588         {
     589             GProperty<float>* _EFFICIENCY = src->getProperty(EFFICIENCY);
     590             assert(_EFFICIENCY && os && "sensor surfaces must have an efficiency" );
     591 
     592             if(m_fake_efficiency >= 0.f && m_fake_efficiency <= 1.0f)
     593             {
     594                 _detect           = makeConstantProperty(m_fake_efficiency) ;
     595                 _absorb           = makeConstantProperty(1.0-m_fake_efficiency);
     596                 _reflect_specular = makeConstantProperty(0.0);
     597                 _reflect_diffuse  = makeConstantProperty(0.0);
     598             }
     599             else
     600             {
     601                 _detect = _EFFICIENCY ;
     602                 _absorb = GProperty<float>::make_one_minus( _detect );
     603                 _reflect_specular = makeConstantProperty(0.0);
     604                 _reflect_diffuse  = makeConstantProperty(0.0);
     605             }
     606         }
     607         else
     608         {
     609             GProperty<float>* _REFLECTIVITY = src->getProperty(REFLECTIVITY);
     610             assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );
     611 
     612             if(os->isSpecular())
     613             {
     614                 _detect  = makeConstantProperty(0.0) ;
     615                 _reflect_specular = _REFLECTIVITY ;
     616                 _reflect_diffuse  = makeConstantProperty(0.0) ;
     617                 _absorb  = GProperty<float>::make_one_minus(_reflect_specular);
     618             }
     619             else
     620             {
     621                 _detect  = makeConstantProperty(0.0) ;
     622                 _reflect_specular = makeConstantProperty(0.0) ;
     623                 _reflect_diffuse  = _REFLECTIVITY ;
     624                 _absorb  = GProperty<float>::make_one_minus(_reflect_diffuse);
     625             }
     626         }
     627     }




