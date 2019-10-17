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




Check performance with v6 in scan-pf-2 : no significant change with scan-pf-0 
---------------------------------------------------------------------------------

::

    [blyth@localhost opticks]$ profilesmrytab.py scan-pf-2/cvd_1_rtx_1 scan-pf-0/cvd_1_rtx_1
    0
    ProfileSmry FromDict:scan-pf-2:cvd_1_rtx_1 /home/blyth/local/opticks/evtbase/scan-pf-2 1:TITAN_RTX 
    1:TITAN_RTX, RTX ON
              CDeviceBriefAll : 0:TITAN_V 1:TITAN_RTX 
              CDeviceBriefVis : 1:TITAN_RTX 
                      RTXMode : 1 
        NVIDIA_DRIVER_VERSION : 435.21 
                     name       note  av.interv  av.launch  av.overhd :                                             launch :                                                  q 
           cvd_1_rtx_1_1M   MULTIEVT     0.1510     0.1422     1.0623 :   0.1484   0.1406   0.1406   0.1406   0.1406   0.1406   0.1406   0.1406   0.1484   0.1406 :   0.1502   0.1429   0.1433   0.1405   0.1414   0.1406   0.1453   0.1401   0.1445   0.1406 
          cvd_1_rtx_1_10M   MULTIEVT     1.4696     1.4414     1.0196 :   1.4375   1.4375   1.4453   1.4453   1.4375   1.4453   1.4375   1.4453   1.4375   1.4453 :   1.4415   1.4366   1.4403   1.4435   1.4400   1.4396   1.4415   1.4406   1.4411   1.4414 
         cvd_1_rtx_1_100M   MULTIEVT    14.1198    13.8750     1.0176 :  13.8984  13.8750  13.8672  13.8750  13.8750  13.8828  13.8750  13.8828  13.8594  13.8594 :  13.8990  13.8749  13.8673  13.8795  13.8717  13.8795  13.8760  13.8818  13.8580  13.8592 

    1
    ProfileSmry FromDict:scan-pf-0:cvd_1_rtx_1 /home/blyth/local/opticks/evtbase/scan-pf-0 1:TITAN_RTX 
    1:TITAN_RTX, RTX ON
              CDeviceBriefAll : 0:TITAN_V 1:TITAN_RTX 
              CDeviceBriefVis : 1:TITAN_RTX 
                      RTXMode : 1 
        NVIDIA_DRIVER_VERSION : 435.21 
                     name       note  av.interv  av.launch  av.overhd :                                             launch :                                                  q 
           cvd_1_rtx_1_1M   MULTIEVT     0.1484     0.1410     1.0526 :   0.1484   0.1484   0.1367   0.1367   0.1406   0.1406   0.1367   0.1406   0.1367   0.1445 :   0.1482   0.1454   0.1376   0.1376   0.1394   0.1410   0.1374   0.1396   0.1374   0.1417 
          cvd_1_rtx_1_10M   MULTIEVT     1.3338     1.3012     1.0251 :   1.3125   1.2969   1.3008   1.3047   1.2969   1.3008   1.3008   1.3008   1.2969   1.3008 :   1.3111   1.2973   1.3008   1.3020   1.2983   1.3010   1.3007   1.3013   1.2993   1.3005 
          cvd_1_rtx_1_20M   MULTIEVT     2.7539     2.6914     1.0232 :   2.6914   2.6914   2.6914   2.6914   2.6875   2.6914   2.6914   2.6914   2.6992   2.6875 :   2.6911   2.6917   2.6908   2.6915   2.6887   2.6920   2.6905   2.6915   2.6999   2.6864 
          cvd_1_rtx_1_30M   MULTIEVT     3.9709     3.8840     1.0224 :   3.8906   3.8789   3.8789   3.8789   3.8789   3.8789   3.8867   3.8867   3.8945   3.8867 :   3.8905   3.8773   3.8795   3.8803   3.8785   3.8800   3.8862   3.8863   3.8951   3.8874 
          cvd_1_rtx_1_40M   MULTIEVT     5.5799     5.4641     1.0212 :   5.4688   5.4609   5.4609   5.4609   5.4648   5.4648   5.4688   5.4648   5.4648   5.4609 :   5.4659   5.4614   5.4627   5.4590   5.4637   5.4666   5.4686   5.4645   5.4667   5.4608 
          cvd_1_rtx_1_50M   MULTIEVT     6.8034     6.6617     1.0213 :   6.6680   6.6523   6.6602   6.6641   6.6641   6.6562   6.6641   6.6641   6.6602   6.6641 :   6.6664   6.6523   6.6607   6.6614   6.6616   6.6583   6.6639   6.6621   6.6607   6.6617 
          cvd_1_rtx_1_60M   MULTIEVT     7.9128     7.7133     1.0259 :   7.7070   7.7070   7.7070   7.7070   7.7109   7.7148   7.7188   7.7227   7.7188   7.7188 :   7.7094   7.7066   7.7067   7.7084   7.7119   7.7147   7.7161   7.7242   7.7193   7.7155 
          cvd_1_rtx_1_70M   MULTIEVT     9.8299     9.6313     1.0206 :   9.6250   9.6211   9.6289   9.6289   9.6289   9.6328   9.6328   9.6328   9.6406   9.6406 :   9.6253   9.6238   9.6268   9.6267   9.6291   9.6325   9.6333   9.6343   9.6391   9.6388 
          cvd_1_rtx_1_80M   MULTIEVT    10.9939    10.7445     1.0232 :  10.7344  10.7305  10.7422  10.7344  10.7461  10.7461  10.7500  10.7539  10.7539  10.7539 :  10.7341  10.7301  10.7440  10.7370  10.7465  10.7468  10.7476  10.7510  10.7540  10.7533 
          cvd_1_rtx_1_90M   MULTIEVT    11.8151    11.5301     1.0247 :  11.5234  11.5195  11.5234  11.5312  11.5234  11.5273  11.5391  11.5391  11.5391  11.5352 :  11.5212  11.5188  11.5211  11.5315  11.5232  11.5288  11.5385  11.5381  11.5396  11.5352 
         cvd_1_rtx_1_100M   MULTIEVT    13.4609    13.1328     1.0250 :  13.1289  13.1211  13.1289  13.1250  13.1367  13.1289  13.1367  13.1406  13.1445  13.1367 :  13.1279  13.1242  13.1311  13.1262  13.1369  13.1316  13.1338  13.1379  13.1437  13.1376 
         cvd_1_rtx_1_200M   MULTIEVT    27.3116    26.6652     1.0242 :  26.5547  26.5664  27.2344  26.6250  26.5977  26.6094  26.6055  26.6250  26.6211  26.6133 :  26.5538  26.5662  27.2344  26.6236  26.5970  26.6101  26.6050  26.6222  26.6188  26.6132 

    scan-pf-2/cvd_1_rtx_1
    scan-pf-0/cvd_1_rtx_1
    [blyth@localhost opticks]$ 




