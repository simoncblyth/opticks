missing-SD-again-FIXED 
===========================


Issue : no SD for JUNO 
----------------------------


::
 
    unset OPTICKS_KEY
    export OPTICKS_KEY=$(geocache-;geocache-j1808-v5-key)

    OKTest    ## and look for SD in the histories, none there

    
FIX
-------

* fixed by changed isSensor to be based on EFFICIENCY prop
  and remaking the geocache into v6 

::

    unset OPTICKS_KEY # not necessary, but makes logging less confusing 

    kcd  # check no key 

    geocache-; geocache-j1808-v6

Above creates the v6 geocache with runcomment::
 
     same-gdml-as-v5-but-fixes-lack-of-SD-with-isSensor-based-on-surface-EFFICIENCY


Check v6
--------------

In setup::

    unset OPTICKS_KEY
    export OPTICKS_KEY=$(geocache-;geocache-j1808-v6-key)    # with sensor surfaces operational allowing SD


Pickup new env::

    ini
    kcd  # check can find the geocache 

    OKTest    ## now SD shows up

    
    OKTest --xanalytic -g -10 --rngmax 10 --rtx 1 --cvd 1     ## higher stats 





lvsdname no longer needed, with GDML handling surface props
-------------------------------------------------------------------

::

   cd notes/issues
   grep lvsdname *.rst



Mint geocache-j1808-v6 to look into this with extradigest to avoid touching v5 cache
---------------------------------------------------------------------------------------

::

   geocache-;geocache-j1808-v6

::

    [blyth@localhost GNodeLib]$ grep inch_inner1_log LVNames.txt | sort  | uniq
    PMT_20inch_inner1_log0x4cb3cc0
    PMT_3inch_inner1_log0x510bb00


* tried creating geocache with "--lvsdname inch_inner1_log" this 
  is handled in X4PhysicalVolume::convertSensors_r which 

::

    X4PhysicalVolume=FATAL geocache-j1808-v6


::

    2019-10-17 16:20:11.278 FATAL [123069] [X4PhysicalVolume::convertSensors_r@298]  is_lvsdname 1 is_sd 0 sdn SD? name PMT_20inch_inner1_log0x4cb3cc0 nameref PMT_20inch_inner1_log0x4cb3cc00x1db38a0
    2019-10-17 16:20:11.278 FATAL [123069] [X4PhysicalVolume::convertSensors_r@298]  is_lvsdname 1 is_sd 0 sdn SD? name PMT_20inch_inner1_log0x4cb3cc0 nameref PMT_20inch_inner1_log0x4cb3cc00x1db38a0
    2019-10-17 16:20:11.278 FATAL [123069] [X4PhysicalVolume::convertSensors_r@298]  is_lvsdname 1 is_sd 0 sdn SD? name PMT_20inch_inner1_log0x4cb3cc0 nameref PMT_20inch_inner1_log0x4cb3cc00x1db38a0
    2019-10-17 16:20:11.278 FATAL [123069] [X4PhysicalVolume::convertSensors_r@298]  is_lvsdname 1 is_sd 0 sdn SD? name PMT_20inch_inner1_log0x4cb3cc0 nameref PMT_20inch_inner1_log0x4cb3cc00x1db38a0



Inkling of the cause
-----------------------

* there are already surfaces present on the cathode, so x4 
  doesnt stuff the special sensor surfaces there

  * so its historical

::

    [blyth@localhost GItemList]$ cat GSurfaceLib.txt
    UpperChimneyTyvekSurface
    PMT_20inch_photocathode_logsurf1
    PMT_20inch_mirror_logsurf1
    PMT_20inch_photocathode_logsurf2
    PMT_3inch_photocathode_logsurf1
    PMT_3inch_absorb_logsurf1
    PMT_3inch_photocathode_logsurf2
    PMT_3inch_absorb_logsurf3
    CDTyvekSurface
    Steel_surface
    Tube_surf
    perfectDetectSurface
    perfectAbsorbSurface
    perfectSpecularSurface
    perfectDiffuseSurface
    [blyth@localhost GItemList]$ 



hookupSD assert : double 0x-refing ?
---------------------------------------

* this was from a revert to runing OKG4Test from the o.sh install

::

    2019-10-17 16:28:43.901 INFO  [143850] [CGDMLDetector::standardizeGeant4MaterialProperties@242] ]
    2019-10-17 16:28:43.901 ERROR [143850] [CDetector::attachSurfaces@350]  some surfaces were found : so assume there is nothing to do 
    2019-10-17 16:28:43.901 ERROR [143850] [CDetector::hookupSD@170] SetSensitiveDetector lvn PMT_20inch_inner1_log0x4cb3cc00x1db38a0 sdn SD? lv 0
    2019-10-17 16:28:43.901 FATAL [143850] [CDetector::hookupSD@177]  no lv PMT_20inch_inner1_log0x4cb3cc00x1db38a0
    OKG4Test: /home/blyth/opticks/cfg4/CDetector.cc:178: void CDetector::hookupSD(): Assertion `lv' failed.
    /home/blyth/local/opticks/bin/o.sh: line 254: 143850 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okx4test --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v5.gdml --digestextra v6 --lvsdname inch_inner1_log --csgskiplv 22 --runfolder geocache-j1808-v6 --runcomment same-gdml-as-v5-but-varies-conversion-arguments
    === o-main : /home/blyth/local/opticks/lib/OKG4Test --okx4test --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v5.gdml --digestextra v6 --lvsdname inch_inner1_log --csgskiplv 22 --runfolder geocache-j1808-v6 --runcomment same-gdml-as-v5-but-varies-conversion-arguments ======= PWD /tmp/blyth/opticks/geocache-create- RC 134 Thu Oct 17 16:28:45 CST 2019
    echo o-postline : dummy
    o-postline : dummy
    /home/blyth/local/opticks/bin/o.sh : RC : 134
    [blyth@localhost 1]$ 


::

    GSurfaceLib=INFO geocache-j1808-v6
    ...
    2019-10-17 16:57:48.882 INFO  [206098] [X4PhysicalVolume::convertSurfaces@366]  num_lbs 9 num_sks 2
    2019-10-17 16:57:49.456 FATAL [206098] [GGeoSensor::AddSensorSurfaces@89]  require a cathode material to AddSensorSurfaces 
    2019-10-17 16:57:49.456 INFO  [206098] [X4PhysicalVolume::convertSensors@228]  m_lvsdname inch_inner1_log num_lvsd 2 num_clv 2 num_bds 9 num_sks0 2 num_sks1 2


With DYB and ancient GDML deficiencies it was expedient to carry the EFFICIENY property 
in Bialkali material and copy it across from the material to fake SensorSurfaces that were added. 
But that doesnt fly for JUNO as it already has surfaces on the cathode and no Bialkali material.

* need to translate the G4 association between the surfaces and the sensitive detector into GGeo model 
  or maybe can just setSensor on the appropriate surfaces ? 

::

    PMT_20inch_photocathode_logsurf1
    PMT_20inch_photocathode_logsurf2
    PMT_3inch_photocathode_logsurf1
    PMT_3inch_photocathode_logsurf2


::

    084 void GGeoSensor::AddSensorSurfaces( GGeo* gg )
     85 {
     86     GMaterial* cathode_props = gg->getCathode() ;
     87     if(!cathode_props)
     88     {
     89         LOG(fatal) << " require a cathode material to AddSensorSurfaces " ;
     90         return ;
     91     }
     92     assert( cathode_props );
     93 
     94     unsigned nclv = gg->getNumCathodeLV();
     95 
     96 
     97     if(nclv == 0)
     98     {
     99         LOG(error) << "NO CathodeLV : so not adding any GSkinSurface to translate sensitivity between models " ;
    100     }
    101 
    102 
    103     for(unsigned i=0 ; i < nclv ; i++)
    104     {
    105         const char* sslv = gg->getCathodeLV(i);
    106 
    107         unsigned num_mat = gg->getNumMaterials()  ;
    108         unsigned num_sks = gg->getNumSkinSurfaces() ;
    109         unsigned num_bds = gg->getNumBorderSurfaces() ;
    110 
    111         unsigned index = num_mat + num_sks + num_bds ;
    112         // standard materials/surfaces use the originating aiMaterial index, 
    113         // extend that for fake SensorSurface by toting up all 
    114 
    115         LOG(LEVEL)
    116                   << " i " << i
    117                   << " sslv " << sslv
    118                   << " index " << index
    119                   << " num_mat " << num_mat
    120                   << " num_sks " << num_sks
    121                   << " num_bds " << num_bds
    122                   ;
    123 
    124         GSkinSurface* gss = MakeSensorSurface(sslv, index);
    125         gss->setStandardDomain();  // default domain 
    126         gss->setSensor();
    127         gss->add(cathode_props);
    128 
    129         LOG(LEVEL) << " gss " << gss->description();
    130 
    131         gg->add(gss);
    132 
    133         {
    134             // not setting sensor or domain : only the standardized need those
    135             GSkinSurface* gss_raw = MakeSensorSurface(sslv, index);
    136             gss_raw->add(cathode_props);
    137             gg->addRaw(gss_raw);
    138         }
    139     }
    140 }





Check on properties of the surfaces::

    GSurfaceLib=INFO X4MaterialPropertiesTable=FATAL geocache-j1808-v6


    2019-10-17 19:04:36.390 INFO  [414139] [X4PhysicalVolume::convertMaterials@343]  num_materials 17
    2019-10-17 19:04:36.390 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name UpperChimneyTyvekSurface         pv1 pUpperChimneyLS0x5b2f160        pv2 pUpperChimneyTyvek0x5b2f300     keys REFLECTIVITY

    2019-10-17 19:04:36.391 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name PMT_20inch_photocathode_logsurf1 pv1 PMT_20inch_inner1_phys0x4c9a870 pv2 PMT_20inch_body_phys0x4c9a7f0   keys RINDEX REFLECTIVITY EFFICIENCY GROUPVEL KINDEX THICKNESS
    2019-10-17 19:04:36.391 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name PMT_20inch_photocathode_logsurf2 pv1 PMT_20inch_body_phys0x4c9a7f0   pv2 PMT_20inch_inner1_phys0x4c9a870 keys RINDEX REFLECTIVITY EFFICIENCY GROUPVEL KINDEX THICKNESS

    2019-10-17 19:04:36.391 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name PMT_3inch_photocathode_logsurf1  pv1 PMT_3inch_inner1_phys0x510beb0  pv2 PMT_3inch_body_phys0x510be30    keys RINDEX REFLECTIVITY EFFICIENCY GROUPVEL KINDEX THICKNESS
    2019-10-17 19:04:36.391 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name PMT_3inch_photocathode_logsurf2  pv1 PMT_3inch_body_phys0x510be30    pv2 PMT_3inch_inner1_phys0x510beb0  keys RINDEX REFLECTIVITY EFFICIENCY GROUPVEL KINDEX THICKNESS


    2019-10-17 19:04:36.391 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name PMT_3inch_absorb_logsurf3        pv1 PMT_3inch_cntr_phys0x510c010    pv2 PMT_3inch_body_phys0x510be30    keys REFLECTIVITY
    2019-10-17 19:04:36.392 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name CDTyvekSurface                   pv1 pOuterWaterPool0x4bd2b70        pv2 pCentralDetector0x4bd4930       keys REFLECTIVITY
    2019-10-17 19:04:36.392 INFO  [414139] [GSurfaceLib::add@409]  GSkinSurface  name Steel_surface keys REFLECTIVITY

    2019-10-17 19:04:36.391 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name PMT_20inch_mirror_logsurf1       pv1 PMT_20inch_inner2_phys0x4c9a920 pv2 PMT_20inch_body_phys0x4c9a7f0   keys REFLECTIVITY
    2019-10-17 19:04:36.391 INFO  [414139] [GSurfaceLib::add@343]  GBorderSurface  name PMT_3inch_absorb_logsurf1        pv1 PMT_3inch_inner2_phys0x510bf60  pv2 PMT_3inch_body_phys0x510be30    keys REFLECTIVITY



Change GPropertyMap::isSensor to be based on EFFICIENCY property that  m_sensor switch, when OLD_SENSOR macro not defined.

* This regained the SD, without the need for kludge adding of SensorSurfaces.



