sensors-review
=================

What is needed
----------------

* DONE : contiguous 0-based sensor_index 

* DONE : sensor data array (float4/uint4) collected in G4Opticks:setSensorData containing:

  * pmt_id  
  * pmt_category (as used to pick appropriate layer of theta texture) 
  * qe float
  * any other floats needed to recreate de (hit culling detector efficiency) in GPU

* 1-to-1 mapping : pmtid <-> sensor_index
* automated way to define the sensor_index and have it provided by GVolume::getIdentity 
  for inclusion into instanceIdentity buffer

* minimize detector specific code

  * Geant4 to Opticks geometry translation needs to auto-identify and collect sensors float4/uint4 
    leaving sensor_index associated with GVolume as well as the Geant4 copyNumber/replicaNumber  

  * need generic Opticks interface to add detector specific extra info into 
    those auto defined base sensors: eg pmt_category, qe other efficiency floats  


DONE : added auto-determined contiguous 0-based sensorIndex 
--------------------------------------------------------------

Invoked from X4PhysicalVolume::convertNode::

    1335     int sensorIndex = m_blib->isSensorBoundary(boundary) ? m_ggeo->addSensorVolume(volume) : -1 ;
    1336     if(sensorIndex > -1) m_blib->countSensorBoundary(boundary);


How to auto-detect sensors ?
-----------------------------

* are using surface with non-zero efficiency, see GPropertyMap::isSensor


geocache-tds dumping volumes with boundary with surfaces
------------------------------------------------------------

::

    geocache-tds -V
    ...

    2020-07-25 15:40:36.238 INFO  [13760730] [GSurfaceLib::dumpSkinSurface@1375] dumpSkinSurface
    2020-07-25 15:40:36.239 INFO  [13760730] [GSurfaceLib::dumpSkinSurface@1380]  SS    0 :                            Steel_surface : lLowerChimneySteel0x47c2280
    2020-07-25 15:40:38.922 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   13 omat/osur/isur/imat (   14  14  -1   2 )  boundaryName vetoWater/CDTyvekSurface//Tyvek
    2020-07-25 15:40:39.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   24 omat/osur/isur/imat (   12   3   1  11 )  boundaryName Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   25 omat/osur/isur/imat (   12  -1   2  11 )  boundaryName Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.046 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   26 omat/osur/isur/imat (   12   6   4  11 )  boundaryName Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.046 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   27 omat/osur/isur/imat (   12  -1   5  11 )  boundaryName Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.046 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   24 omat/osur/isur/imat (   12   3   1  11 )  boundaryName Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.047 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   25 omat/osur/isur/imat (   12  -1   2  11 )  boundaryName Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.047 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   24 omat/osur/isur/imat (   12   3   1  11 )  boundaryName Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.047 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   25 omat/osur/isur/imat (   12  -1   2  11 )  boundaryName Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.047 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   26 omat/osur/isur/imat (   12   6   4  11 )  boundaryName Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.048 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   27 omat/osur/isur/imat (   12  -1   5  11 )  boundaryName Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.048 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   24 omat/osur/isur/imat (   12   3   1  11 )  boundaryName Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.048 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   25 omat/osur/isur/imat (   12  -1   2  11 )  boundaryName Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.049 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   24 omat/osur/isur/imat (   12   3   1  11 )  boundaryName Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.049 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   25 omat/osur/isur/imat (   12  -1   2  11 )  boundaryName Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.049 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   24 omat/osur/isur/imat (   12   3   1  11 )  boundaryName Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.049 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   25 omat/osur/isur/imat (   12  -1   2  11 )  boundaryName Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:39.050 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   26 omat/osur/isur/imat (   12   6   4  11 )  boundaryName Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:39.050 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   27 omat/osur/isur/imat (   12  -1   5  11 )  boundaryName Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum
    ...
    2020-07-25 15:40:47.044 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   26 omat/osur/isur/imat (   12   6   4  11 )  boundaryName Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:47.044 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   27 omat/osur/isur/imat (   12  -1   5  11 )  boundaryName Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:47.044 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   24 omat/osur/isur/imat (   12   3   1  11 )  boundaryName Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:47.044 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   25 omat/osur/isur/imat (   12  -1   2  11 )  boundaryName Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum
    2020-07-25 15:40:47.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   28 omat/osur/isur/imat (   12   9   7  11 )  boundaryName Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:47.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   29 omat/osur/isur/imat (   12  -1   8  11 )  boundaryName Pyrex//PMT_3inch_absorb_logsurf1/Vacuum
    2020-07-25 15:40:47.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   28 omat/osur/isur/imat (   12   9   7  11 )  boundaryName Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:47.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   29 omat/osur/isur/imat (   12  -1   8  11 )  boundaryName Pyrex//PMT_3inch_absorb_logsurf1/Vacuum
    2020-07-25 15:40:47.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   28 omat/osur/isur/imat (   12   9   7  11 )  boundaryName Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:47.045 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   29 omat/osur/isur/imat (   12  -1   8  11 )  boundaryName Pyrex//PMT_3inch_absorb_logsurf1/Vacuum
    ...
    2020-07-25 15:40:56.222 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   33 omat/osur/isur/imat (   12  13  11  11 )  boundaryName Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:56.222 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   34 omat/osur/isur/imat (   12  -1  12  11 )  boundaryName Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum
    2020-07-25 15:40:56.222 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   33 omat/osur/isur/imat (   12  13  11  11 )  boundaryName Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:56.222 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   34 omat/osur/isur/imat (   12  -1  12  11 )  boundaryName Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum
    2020-07-25 15:40:56.222 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   33 omat/osur/isur/imat (   12  13  11  11 )  boundaryName Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
    2020-07-25 15:40:56.223 INFO  [13760730] [*X4PhysicalVolume::convertNode@1194]  copyNumber        0 boundary   34 omat/osur/isur/imat (   12  -1  12  11 )  boundaryName Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum


1. find surfaces with non-zero efficiency properties 
2. find boundaries with those sensor surfaces 

   * THIS DOESNT FLY AS WANT TO ASSIGN SENSOR_INDEX IN SAME TRAVERSE THAT IS COLLECTING BOUNDARIES
     SO OPERATE WITHIN THE SURFACE ONLY 

3. collect volumes with those boundaries into m_sensor_volumes and assign sensor_index to the volumes  







::

    2020-07-25 16:26:58.569 INFO  [13803875] [GSurfaceLib::dumpSurfaces@727] X4PhysicalVolume::convertSurfaces num_surfaces 20
     index :  0 is_sensor : N type :        bordersurface name :                           UpperChimneyTyvekSurface bpv1 pUpperChimneyLS0x47bfd70 bpv2 pUpperChimneyTyvek0x47bfed0 .
     index :  1 is_sensor : Y type :        bordersurface name :        NNVTMCPPMT_PMT_20inch_photocathode_logsurf1 bpv1 NNVTMCPPMT_PMT_20inch_inner1_phys0x35695e0 bpv2 NNVTMCPPMT_PMT_20inch_body_phys0x3569550 .
     index :  2 is_sensor : N type :        bordersurface name :              NNVTMCPPMT_PMT_20inch_mirror_logsurf1 bpv1 NNVTMCPPMT_PMT_20inch_inner2_phys0x35696a0 bpv2 NNVTMCPPMT_PMT_20inch_body_phys0x3569550 .
     index :  3 is_sensor : Y type :        bordersurface name :        NNVTMCPPMT_PMT_20inch_photocathode_logsurf2 bpv1 NNVTMCPPMT_PMT_20inch_body_phys0x3569550 bpv2 NNVTMCPPMT_PMT_20inch_inner1_phys0x35695e0 .
     index :  4 is_sensor : Y type :        bordersurface name :   HamamatsuR12860_PMT_20inch_photocathode_logsurf1 bpv1 HamamatsuR12860_PMT_20inch_inner1_phys0x35482b0 bpv2 HamamatsuR12860_PMT_20inch_body_phys0x3548210 .
     index :  5 is_sensor : N type :        bordersurface name :         HamamatsuR12860_PMT_20inch_mirror_logsurf1 bpv1 HamamatsuR12860_PMT_20inch_inner2_phys0x3548380 bpv2 HamamatsuR12860_PMT_20inch_body_phys0x3548210 .
     index :  6 is_sensor : Y type :        bordersurface name :   HamamatsuR12860_PMT_20inch_photocathode_logsurf2 bpv1 HamamatsuR12860_PMT_20inch_body_phys0x3548210 bpv2 HamamatsuR12860_PMT_20inch_inner1_phys0x35482b0 .
     index :  7 is_sensor : Y type :        bordersurface name :                    PMT_3inch_photocathode_logsurf1 bpv1 PMT_3inch_inner1_phys0x3ceb800 bpv2 PMT_3inch_body_phys0x3ceb780 .
     index :  8 is_sensor : N type :        bordersurface name :                          PMT_3inch_absorb_logsurf1 bpv1 PMT_3inch_inner2_phys0x3ceb8b0 bpv2 PMT_3inch_body_phys0x3ceb780 .
     index :  9 is_sensor : Y type :        bordersurface name :                    PMT_3inch_photocathode_logsurf2 bpv1 PMT_3inch_body_phys0x3ceb780 bpv2 PMT_3inch_inner1_phys0x3ceb800 .
     index : 10 is_sensor : N type :        bordersurface name :                          PMT_3inch_absorb_logsurf3 bpv1 PMT_3inch_cntr_phys0x3ceb960 bpv2 PMT_3inch_body_phys0x3ceb780 .
     index : 11 is_sensor : Y type :        bordersurface name :              PMT_20inch_veto_photocathode_logsurf1 bpv1 PMT_20inch_veto_inner1_phys0x355b8f0 bpv2 PMT_20inch_veto_body_phys0x355b870 .
     index : 12 is_sensor : N type :        bordersurface name :                    PMT_20inch_veto_mirror_logsurf1 bpv1 PMT_20inch_veto_inner2_phys0x355b9a0 bpv2 PMT_20inch_veto_body_phys0x355b870 .
     index : 13 is_sensor : Y type :        bordersurface name :              PMT_20inch_veto_photocathode_logsurf2 bpv1 PMT_20inch_veto_body_phys0x355b870 bpv2 PMT_20inch_veto_inner1_phys0x355b8f0 .
     index : 14 is_sensor : N type :        bordersurface name :                                     CDTyvekSurface bpv1 pOuterWaterPool0x339c960 bpv2 pCentralDetector0x339e6d0 .
     index : 15 is_sensor : N type :          skinsurface name :                                      Steel_surface sslv lLowerChimneySteel0x47c2280 .
     index : 16 is_sensor : Y type :          testsurface name :                               perfectDetectSurface .
     index : 17 is_sensor : N type :          testsurface name :                               perfectAbsorbSurface .
     index : 18 is_sensor : N type :          testsurface name :                             perfectSpecularSurface .
     index : 19 is_sensor : N type :          testsurface name :                              perfectDiffuseSurface .


Note that the is_sensor surfaces come in bordersurface swapped volume pairs:: 

     index :  1 is_sensor : Y type :        bordersurface name :        NNVTMCPPMT_PMT_20inch_photocathode_logsurf1 bpv1 NNVTMCPPMT_PMT_20inch_inner1_phys0x35695e0 bpv2 NNVTMCPPMT_PMT_20inch_body_phys0x3569550 .
     index :  3 is_sensor : Y type :        bordersurface name :        NNVTMCPPMT_PMT_20inch_photocathode_logsurf2 bpv1 NNVTMCPPMT_PMT_20inch_body_phys0x3569550 bpv2 NNVTMCPPMT_PMT_20inch_inner1_phys0x35695e0 .

     index :  4 is_sensor : Y type :        bordersurface name :   HamamatsuR12860_PMT_20inch_photocathode_logsurf1 bpv1 HamamatsuR12860_PMT_20inch_inner1_phys0x35482b0 bpv2 HamamatsuR12860_PMT_20inch_body_phys0x3548210 .
     index :  6 is_sensor : Y type :        bordersurface name :   HamamatsuR12860_PMT_20inch_photocathode_logsurf2 bpv1 HamamatsuR12860_PMT_20inch_body_phys0x3548210 bpv2 HamamatsuR12860_PMT_20inch_inner1_phys0x35482b0 .

     index :  7 is_sensor : Y type :        bordersurface name :                    PMT_3inch_photocathode_logsurf1 bpv1 PMT_3inch_inner1_phys0x3ceb800 bpv2 PMT_3inch_body_phys0x3ceb780 .
     index :  9 is_sensor : Y type :        bordersurface name :                    PMT_3inch_photocathode_logsurf2 bpv1 PMT_3inch_body_phys0x3ceb780 bpv2 PMT_3inch_inner1_phys0x3ceb800 .

     index : 11 is_sensor : Y type :        bordersurface name :              PMT_20inch_veto_photocathode_logsurf1 bpv1 PMT_20inch_veto_inner1_phys0x355b8f0 bpv2 PMT_20inch_veto_body_phys0x355b870 .
     index : 13 is_sensor : Y type :        bordersurface name :              PMT_20inch_veto_photocathode_logsurf2 bpv1 PMT_20inch_veto_body_phys0x355b870 bpv2 PMT_20inch_veto_inner1_phys0x355b8f0 .

     index : 16 is_sensor : Y type :          testsurface name :                               perfectDetectSurface .


     Added this: 

     2020-07-25 17:22:45.104 INFO  [13851512] [GPropertyLib::dumpSensorIndices@935] X4PhysicalVolume::convertSurfaces  NumSensorIndices 9 ( 1 3 4 6 7 9 11 13 16  ) 



     index :  0 is_sensor : N type :        bordersurface name :                           UpperChimneyTyvekSurface bpv1 pUpperChimneyLS0x47bfd70 bpv2 pUpperChimneyTyvek0x47bfed0 .
     index :  2 is_sensor : N type :        bordersurface name :              NNVTMCPPMT_PMT_20inch_mirror_logsurf1 bpv1 NNVTMCPPMT_PMT_20inch_inner2_phys0x35696a0 bpv2 NNVTMCPPMT_PMT_20inch_body_phys0x3569550 .
     index :  5 is_sensor : N type :        bordersurface name :         HamamatsuR12860_PMT_20inch_mirror_logsurf1 bpv1 HamamatsuR12860_PMT_20inch_inner2_phys0x3548380 bpv2 HamamatsuR12860_PMT_20inch_body_phys0x3548210 .
     index :  8 is_sensor : N type :        bordersurface name :                          PMT_3inch_absorb_logsurf1 bpv1 PMT_3inch_inner2_phys0x3ceb8b0 bpv2 PMT_3inch_body_phys0x3ceb780 .
     index : 10 is_sensor : N type :        bordersurface name :                          PMT_3inch_absorb_logsurf3 bpv1 PMT_3inch_cntr_phys0x3ceb960 bpv2 PMT_3inch_body_phys0x3ceb780 .
     index : 12 is_sensor : N type :        bordersurface name :                    PMT_20inch_veto_mirror_logsurf1 bpv1 PMT_20inch_veto_inner2_phys0x355b9a0 bpv2 PMT_20inch_veto_body_phys0x355b870 .
     index : 14 is_sensor : N type :        bordersurface name :                                     CDTyvekSurface bpv1 pOuterWaterPool0x339c960 bpv2 pCentralDetector0x339e6d0 .
     index : 15 is_sensor : N type :          skinsurface name :                                      Steel_surface sslv lLowerChimneySteel0x47c2280 .
     index : 17 is_sensor : N type :          testsurface name :                               perfectAbsorbSurface .
     index : 18 is_sensor : N type :          testsurface name :                             perfectSpecularSurface .
     index : 19 is_sensor : N type :          testsurface name :                              perfectDiffuseSurface .


::

     288 template <class T>
     289 bool GPropertyMap<T>::isSensor()
     290 {
     291 #ifdef OLD_SENSOR
     292     return m_sensor ;
     293 #else
     294     return hasNonZeroProperty(EFFICIENCY) || hasNonZeroProperty(detect) ;
     295 #endif
     296 }
     297 template <class T>
     298 void GPropertyMap<T>::setSensor(bool sensor)
     299 {
     300 #ifdef OLD_SENSOR
     301     m_sensor = sensor ;
     302 #else
     303     assert(0 && "sensors are now detected by the prescense of an EFFICIENCY property" );
     304 #endif
     305 }



Added this::

    2020-07-26 10:11:51.891 INFO  [14283890] [X4PhysicalVolume::convertStructure@919] ] GGeo::getNumVolumes() 316326 GGeo::getNumSensorVolumes() 45612
     GGeo::getSensorBoundaryReport() 
     boundary  24 b+1  25 sensor_count  12612 Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
     boundary  26 b+1  27 sensor_count   5000 Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum
     boundary  28 b+1  29 sensor_count  25600 Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
     boundary  33 b+1  34 sensor_count   2400 Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
                          sensor_total  45612






Checking on a surface G4OpBoundaryProcess::PostStepDoIt
---------------------------------------------------------

::

     318         G4LogicalSurface* Surface = NULL;
     319 
     320         Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
     321 
     322         if (Surface == NULL){
     323           G4bool enteredDaughter= (thePostPV->GetMotherLogical() ==
     324                                    thePrePV ->GetLogicalVolume());
     325       if(enteredDaughter){
     326         Surface =
     327               G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     328         if(Surface == NULL)
     329           Surface =
     330                 G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     331       }
     332       else {
     333         Surface =
     334               G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     335         if(Surface == NULL)
     336           Surface =
     337                 G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     338       }
     339     }
     340 
     341         if (Surface) OpticalSurface =
     342            dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
     343 
     344         if (OpticalSurface) {
     345 
     346            type      = OpticalSurface->GetType();
     347            theModel  = OpticalSurface->GetModel();
     348            theFinish = OpticalSurface->GetFinish();
     349 
     350            aMaterialPropertiesTable = OpticalSurface->
     351                                         GetMaterialPropertiesTable();
     352 
     353            if (aMaterialPropertiesTable) {
     ...
     ...            RINDEX, RELECTIVITY
     ...
     391               PropertyPointer =
     392               aMaterialPropertiesTable->GetProperty(kEFFICIENCY);
     393               if (PropertyPointer) {
     394                       theEfficiency =
     395                       PropertyPointer->Value(thePhotonMomentum);
     396               }
     397 




Earlier (1st?) approach : NSensors
-------------------------------------

Old Dead Code, AssimpGGeo::convertSensorsVisit

* required an input sensor list that associated node indices with sensors 

  * that is a brittle approach : as node indices change too much 


::

     727     NSensor* sensor = m_sensor_list ? m_sensor_list->findSensorForNode( nodeIndex ) : NULL ;
     728 
     729     //const char* sd = "SD_AssimpGGeo" ; 
     730     const char* sd = "SD0" ;
     731 
     732 
     733 #ifdef OLD_CATHODE
     734     GMaterial* cathode = gg->getCathode() ;
     735 
     736     const char* cathode_material_name = gg->getCathodeMaterialName() ;
     737     bool name_match = strcmp(mt_name, cathode_material_name) == 0 ;
     738     bool ptr_match = mt == cathode ;   // <--- always false 
     739 
     740     if(sensor && name_match)
     741     {
     742          LOG(debug) << "AssimpGGeo::convertSensorsVisit "
     743                    << " depth " << depth
     744                    << " lv " << lv
     745                    << " sd " << sd
     746                    << " ptr_match " << ptr_match
     747                    ;
     748          gg->addLVSD(lv, sd) ;
     749     }
     750 
     751 #else
     752     if(sensor)
     753     {
     754         gg->addLVSDMT(lv, sd, mt_name) ;
     755     }
     756 
     757 #endif



What enables Opticks to yield SURFACE_DETECT hits ?
--------------------------------------------------------

* optical buffer needs to have non-zero index for the surface
  and it must have non-zero detect property   

* :doc:`requirements-for-SURFACE_DETECT`





