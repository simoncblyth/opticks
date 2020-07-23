material-inconsistency
=========================

The GItemList/GMaterialLib.txt from geocache-tds (booted from GDML) 
has fewer materials than the original one from tds (booted from in memory Geant4 geometry). 

**Because GDML is not writing all the materials.**

This is a problem because the meaning of the material indices 
will then be messed up.

tds
    passes Geant4 in memory geometry via G4Opticks 

geocache-tds
    parses GDML file 
     

CAUSE FOUND : GDML only adds **USED** materials in the structure traverse (postorder)
---------------------------------------------------------------------------------------


::

    epsilon:src blyth$ g4-cls G4GDMLWriteStructure
    vi -R source/persistency/gdml/include/G4GDMLWriteStructure.hh source/persistency/gdml/src/G4GDMLWriteStructure.cc
    2 files to edit


    388 G4Transform3D G4GDMLWriteStructure::
    389 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    390 {
    ...
    570    AddMaterial(volumePtr->GetMaterial());
    571    // Add the involved materials and solids!
    572 
    573    AddSolid(solidPtr);
    574 
    575    SkinSurfaceCache(GetSkinSurface(volumePtr));
    576 
    577    return R;
    578 }


tds materials
-----------------


::

    2020-07-23 22:02:16.485 ERROR [286857] [X4MaterialTable::init@88] PROCEEDING TO convert material with no mpt Aluminium
    2020-07-23 22:02:16.485 ERROR [286857] [X4MaterialTable::init@88] PROCEEDING TO convert material with no mpt TiO2
    2020-07-23 22:02:16.485 ERROR [286857] [X4MaterialTable::init@88] PROCEEDING TO convert material with no mpt TiO2Coating
    2020-07-23 22:02:16.487 INFO  [286857] [X4PhysicalVolume::convertMaterials@362]  num_materials 39 num_material_with_efficiency 7
    2020-07-23 22:02:16.487 INFO  [286857] [GMaterialLib::dumpSensitiveMaterials@1225] X4PhysicalVolume::convertMaterials num_sensitive_materials 7
    ..

    2020-07-23 22:03:02.501 INFO  [286857] [NCSG::savesrc@315]  treedir_ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/e7b204fa62c028f3d23c102bc554dcbb/1/GMeshLibNCSG/62

::

    [blyth@localhost 1]$ ~/opticks/bin/cat.py GItemList/GMaterialLib.txt 
    0    1    Galactic
    1    2    LS
    2    3    LAB
    3    4    ESR
    4    5    Tyvek
    5    6    Acrylic
    6    7    DummyAcrylic
    7    8    Teflon
    8    9    Steel
    9    10   StainlessSteel
    10   11   Mylar
    11   12   Copper
    12   13   ETFE
    13   14   FEP
    14   15   PE_PA
    15   16   PA
    16   17   Air
    17   18   Vacuum
    18   19   VacuumT
    19   20   photocathode
    20   21   photocathode_3inch
    21   22   photocathode_MCP20inch
    22   23   photocathode_MCP8inch
    23   24   photocathode_Ham20inch
    24   25   photocathode_Ham8inch
    25   26   photocathode_HZC9inch
    26   27   SiO2
    27   28   B2O2
    28   29   Na2O
    29   30   Pyrex
    30   31   MineralOil
    31   32   Rock
    32   33   vetoWater
    33   34   Water
    34   35   Scintillator
    35   36   Adhesive
    36   37   Aluminium
    37   38   TiO2
    38   39   TiO2Coating
    [blyth@localhost 1]$ pwd
    /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/e7b204fa62c028f3d23c102bc554dcbb/1
    [blyth@localhost 1]$ 


    epsilon:1 blyth$ ~/opticks/bin/cat.py GItemList/GMaterialLib.txt 
    0    1    LS
    1    2    Steel
    2    3    Tyvek
    3    4    Air
    4    5    Scintillator
    5    6    TiO2Coating
    6    7    Adhesive
    7    8    Aluminium
    8    9    Rock
    9    10   Acrylic
    10   11   PE_PA
    11   12   Vacuum
    12   13   Pyrex
    13   14   Water
    14   15   vetoWater
    15   16   Galactic
    epsilon:1 blyth$ pwd
    /usr/local/opticks/geocache/OKX4Test_lWorld0x30d4f90_PV_g4live/g4ok_gltf/ad026c799f5511ddb91eb379efa84bc4/1
    epsilon:1 blyth$ 



::

    OPTICKS_RESOURCE_LAYOUT=2 tds 



gdml materials, only 15 
-------------------------

::

    [blyth@localhost ~]$ grep material\ name= $OPTICKS_TOP/tds_ngt_pcnk_sycg.gdml
        <material name="LS0x332b5d0" state="solid">
        <material name="Steel0x334fbb0" state="solid">
        <material name="Tyvek0x334b2d0" state="solid">
        <material name="Air0x3362e60" state="gas">
        <material name="Scintillator0x3386aa0" state="solid">
        <material name="TiO2Coating0x3388d20" state="solid">
        <material name="Adhesive0x3386de0" state="solid">
        <material name="Rock0x3376380" state="solid">
        <material name="Acrylic0x334ccd0" state="solid">
        <material name="PE_PA0x335cfc0" state="solid">
        <material name="Vacuum0x3364ea0" state="gas">
        <material name="Pyrex0x3377b10" state="solid">
        <material name="Water0x3383410" state="solid">
        <material name="vetoWater0x337fc70" state="solid">
        <material name="Galactic0x332b030" state="gas">
    [blyth@localhost ~]$ 





Note material inconsistency, so grab the GDML (sycg : simplify_csg ):: 

    epsilon:opticks blyth$ scp P:/tmp/tds_ngt_pcnk_sycg.gdml .

::

     342 geocache-tds(){
     343     local msg="=== $FUNCNAME :"
     344 
     345     export GMaterialLib=INFO
     346     export GSurfaceLib=INFO
     347     export X4Solid=INFO
     348     export X4PhysicalVolume=INFO
     349     export GMesh=INFO
     350     export OGeo=INFO
     351 
     352 
     353     #local label=tds
     354     #local label=tds_ngt_pcnk
     355     local label=tds_ngt_pcnk_sycg
     356     local gdml=$(opticks-prefix)/$label.gdml
     357     echo $msg gdml $gdml
     358     geocache-gdml-kludge      $gdml
     359     geocache-gdml-kludge-dump $gdml
     360     
     361     geocache-create- --gdmlpath $gdml -D
     362 
     363 }





::

    epsilon:ana blyth$ ./blib.py $GC
     nbnd  35 nmat  16 nsur  20 
      0 :   1 : Galactic///Galactic 
      1 :   2 : Galactic///Rock 
      2 :   3 : Rock///Air 
      3 :   4 : Air///Air 
      4 :   5 : Air///LS 
      5 :   6 : Air///Steel 
      6 :   7 : Air///Tyvek 
      7 :   8 : Air///Aluminium 
      8 :   9 : Aluminium///Adhesive 
      9 :  10 : Adhesive///TiO2Coating 
     10 :  11 : TiO2Coating///Scintillator 
     11 :  12 : Rock///Tyvek 
     12 :  13 : Tyvek///vetoWater 
     13 :  14 : vetoWater/CDTyvekSurface//Tyvek 
     14 :  15 : Tyvek///Water 
     15 :  16 : Water///Acrylic 
     16 :  17 : Acrylic///LS 
     17 :  18 : LS///Acrylic 
     18 :  19 : LS///PE_PA 
     19 :  20 : Water///Steel 
     20 :  21 : Water///PE_PA 
     21 :  22 : Water///Water 
     22 :  23 : Water///Pyrex 
     23 :  24 : Pyrex///Pyrex 
     24 :  25 : Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum 
     25 :  26 : Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum 
     26 :  27 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 
     27 :  28 : Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum 
     28 :  29 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     29 :  30 : Pyrex//PMT_3inch_absorb_logsurf1/Vacuum 
     30 :  31 : Water///LS 
     31 :  32 : Water/Steel_surface/Steel_surface/Steel 
     32 :  33 : vetoWater///Water 
     33 :  34 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 
     34 :  35 : Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum 
    epsilon:ana blyth$ 


