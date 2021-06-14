GSurfaceLib__addImplicitBorderSurface_RINDEX_NoRINDEX
========================================================

The issue that motivated addition of implicit border surface was a divergence 
between OK and G4 for photons reaching the Tyvek.

* :doc:`tds3gun_nonaligned_comparison`


geocache-jun15
------------------

::

    2021-06-14 14:32:14.453 ERROR [5497401] [X4MaterialTable::init@117] PROCEEDING TO convert material with no mpt TiO2Coating
    2021-06-14 14:32:14.453 ERROR [5497401] [X4MaterialTable::init@117] PROCEEDING TO convert material with no mpt Adhesive
    2021-06-14 14:32:14.453 ERROR [5497401] [X4MaterialTable::init@117] PROCEEDING TO convert material with no mpt Aluminium
    2021-06-14 14:32:14.454 ERROR [5497401] [X4MaterialTable::init@117] PROCEEDING TO convert material with no mpt Galactic
    2021-06-14 14:32:14.455 INFO  [5497401] [X4PhysicalVolume::convertMaterials@303]  used_materials.size 17 num_material_with_efficiency 0
    2021-06-14 14:32:14.455 INFO  [5497401] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 0
    2021-06-14 14:32:14.457 INFO  [5497401] [X4PhysicalVolume::convertImplicitSurfaces_r@397]  parent_mtName Rock daughter_mtName Air
    2021-06-14 14:32:14.457 INFO  [5497401] [X4PhysicalVolume::convertImplicitSurfaces_r@402]  RINDEX_NoRINDEX 1 NoRINDEX_RINDEX 0 pv1              pExpHall0x32b9fa0 pv2              pTopRock0x32b9af0 bs 0x0 no-prior-border-surface-adding-implicit 
    2021-06-14 14:32:14.460 INFO  [5497401] [X4PhysicalVolume::convertImplicitSurfaces_r@397]  parent_mtName Tyvek daughter_mtName vetoWater
    2021-06-14 14:32:14.460 INFO  [5497401] [X4PhysicalVolume::convertImplicitSurfaces_r@402]  RINDEX_NoRINDEX 1 NoRINDEX_RINDEX 0 pv1       pOuterWaterPool0x3356e90 pv2           pPoolLining0x32bf050 bs 0x0 no-prior-border-surface-adding-implicit 
    2021-06-14 14:32:14.461 INFO  [5497401] [X4PhysicalVolume::convertImplicitSurfaces_r@397]  parent_mtName Tyvek daughter_mtName Water
    2021-06-14 14:32:14.461 INFO  [5497401] [X4PhysicalVolume::convertImplicitSurfaces_r@402]  RINDEX_NoRINDEX 1 NoRINDEX_RINDEX 0 pv1           pInnerWater0x3358a70 pv2      pCentralDetector0x3358c60 bs 0x0 no-prior-border-surface-adding-implicit 
    2021-06-14 14:32:14.513 INFO  [5497401] [GSurfaceLib::dumpImplicitBorderSurfaces@755] X4PhysicalVolume::convertSurfaces
     num_implicit_border_surfaces 3
    Implicit_RINDEX_NoRINDEX_pExpHall0x32b9fa0_pTopRock0x32b9af0
    Implicit_RINDEX_NoRINDEX_pOuterWaterPool0x3356e90_pPoolLining0x32bf050
    Implicit_RINDEX_NoRINDEX_pInnerWater0x3358a70_pCentralDetector0x3358c60

    2021-06-14 14:32:14.513 INFO  [5497401] [GSurfaceLib::dumpSurfaces@872] X4PhysicalVolume::convertSurfaces num_surfaces 23
     index :  0 is_sensor : N type :        bordersurface name :                           UpperChimneyTyvekSurface bpv1 pUpperChimneyLS0x4cc9e20 bpv2 pUpperChimneyTyvek0x4cc9fc0 .
     index :  1 is_sensor : Y type :        bordersurface name :                   NNVTMCPPMT_photocathode_logsurf1 bpv1 NNVTMCPPMT_inner1_phys0x3a933a0 bpv2 NNVTMCPPMT_body_phys0x3a93320 .
     index :  2 is_sensor : N type :        bordersurface name :                         NNVTMCPPMT_mirror_logsurf1 bpv1 NNVTMCPPMT_inner2_phys0x3a93450 bpv2 NNVTMCPPMT_body_phys0x3a93320 .
     index :  3 is_sensor : Y type :        bordersurface name :                   NNVTMCPPMT_photocathode_logsurf2 bpv1 NNVTMCPPMT_body_phys0x3a93320 bpv2 NNVTMCPPMT_inner1_phys0x3a933a0 .
     index :  4 is_sensor : Y type :        bordersurface name :              HamamatsuR12860_photocathode_logsurf1 bpv1 HamamatsuR12860_inner1_phys0x3aa0c00 bpv2 HamamatsuR12860_body_phys0x3aa0b80 .
     index :  5 is_sensor : N type :        bordersurface name :                    HamamatsuR12860_mirror_logsurf1 bpv1 HamamatsuR12860_inner2_phys0x3aa0cb0 bpv2 HamamatsuR12860_body_phys0x3aa0b80 .
     index :  6 is_sensor : Y type :        bordersurface name :              HamamatsuR12860_photocathode_logsurf2 bpv1 HamamatsuR12860_body_phys0x3aa0b80 bpv2 HamamatsuR12860_inner1_phys0x3aa0c00 .
     index :  7 is_sensor : Y type :        bordersurface name :                    PMT_3inch_photocathode_logsurf1 bpv1 PMT_3inch_inner1_phys0x421eca0 bpv2 PMT_3inch_body_phys0x421ec20 .
     index :  8 is_sensor : N type :        bordersurface name :                          PMT_3inch_absorb_logsurf1 bpv1 PMT_3inch_inner2_phys0x421ed50 bpv2 PMT_3inch_body_phys0x421ec20 .
     index :  9 is_sensor : Y type :        bordersurface name :                    PMT_3inch_photocathode_logsurf2 bpv1 PMT_3inch_body_phys0x421ec20 bpv2 PMT_3inch_inner1_phys0x421eca0 .
     index : 10 is_sensor : N type :        bordersurface name :                          PMT_3inch_absorb_logsurf3 bpv1 PMT_3inch_cntr_phys0x421ee00 bpv2 PMT_3inch_body_phys0x421ec20 .
     index : 11 is_sensor : Y type :        bordersurface name :              PMT_20inch_veto_photocathode_logsurf1 bpv1 PMT_20inch_veto_inner1_phys0x3a8cf20 bpv2 PMT_20inch_veto_body_phys0x3a8cea0 .
     index : 12 is_sensor : N type :        bordersurface name :                    PMT_20inch_veto_mirror_logsurf1 bpv1 PMT_20inch_veto_inner2_phys0x3a8cfd0 bpv2 PMT_20inch_veto_body_phys0x3a8cea0 .
     index : 13 is_sensor : Y type :        bordersurface name :              PMT_20inch_veto_photocathode_logsurf2 bpv1 PMT_20inch_veto_body_phys0x3a8cea0 bpv2 PMT_20inch_veto_inner1_phys0x3a8cf20 .
     index : 14 is_sensor : N type :        bordersurface name :                                     CDTyvekSurface bpv1 pOuterWaterPool0x3356e90 bpv2 pCentralDetector0x3358c60 .
     index : 15 is_sensor : N type :          skinsurface name :                                      Steel_surface sslv lLowerChimneySteel0x4ccc370 .
     index : 16 is_sensor : N type :        bordersurface name : Implicit_RINDEX_NoRINDEX_pExpHall0x32b9fa0_pTopRock bpv1 pExpHall0x32b9fa0 bpv2 pTopRock0x32b9af0 .
     index : 17 is_sensor : N type :        bordersurface name : Implicit_RINDEX_NoRINDEX_pOuterWaterPool0x3356e90_pPoolLining bpv1 pOuterWaterPool0x3356e90 bpv2 pPoolLining0x32bf050 .
     index : 18 is_sensor : N type :        bordersurface name : Implicit_RINDEX_NoRINDEX_pInnerWater0x3358a70_pCentralDetector bpv1 pInnerWater0x3358a70 bpv2 pCentralDetector0x3358c60 .
     index : 19 is_sensor : Y type :          testsurface name :                               perfectDetectSurface .
     index : 20 is_sensor : N type :          testsurface name :                               perfectAbsorbSurface .
     index : 21 is_sensor : N type :          testsurface name :                             perfectSpecularSurface .
     index : 22 is_sensor : N type :          testsurface name :                              perfectDiffuseSurface .
    2021-06-14 14:32:14.513 INFO  [5497401] [GPropertyLib::dumpSensorIndices@933] X4PhysicalVolume::convertSurfaces  NumSensorIndices 9 ( 1 3 4 6 7 9 11 13 19  ) 
    2021-06-14 14:32:14.514 INFO  [5497401] [BFile::preparePath@836] created directory /usr/local/opticks/geocache/OKX4Test_lWorld0x32a96e0_PV_g4live/g4ok_gltf/a3cbac8189a032341f76682cdb4f47b6/1/g4codegen/tests


