material-name-abbreviations
================================

Auto abbreviator needs to handle lots
of very similar names.

::

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




::

    2020-06-02 22:58:37.659 INFO  [20290] [GMaterialLib::sort@476] ORDER_BY_PREFERENCE
    2020-06-02 22:58:37.660 INFO  [20290] [GMaterialLib::createMeta@521] [
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [Galactic] ab [Ga]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [Galactic] ab [Ga] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [LS] ab [LS]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [LS] ab [LS] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [LAB] ab [LA]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [LAB] ab [LA] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [ESR] ab [ES]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [ESR] ab [ES] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [Tyvek] ab [Ty]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [Tyvek] ab [Ty] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [Acrylic] ab [Ac]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [Acrylic] ab [Ac] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [DummyAcrylic] ab [DA]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [DummyAcrylic] ab [DA] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [Teflon] ab [Te]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [Teflon] ab [Te] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [Steel] ab [St]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [Steel] ab [St] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [StainlessSteel] ab [SS]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [StainlessSteel] ab [SS] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [Mylar] ab [My]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [Mylar] ab [My] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [Copper] ab [Co]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [Copper] ab [Co] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [ETFE] ab [ET]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [ETFE] ab [ET] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [FEP] ab [FE]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [FEP] ab [FE] is_now_free 1
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@63]  name [PE_PA] ab [PE]
    2020-06-02 22:58:37.660 INFO  [20290] [SAbbrev::init@75]  name [PE_PA] ab [PE] is_now_free 1
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@63]  name [PA] ab [PA]
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@75]  name [PA] ab [PA] is_now_free 1
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@63]  name [Air] ab [Ai]
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@75]  name [Air] ab [Ai] is_now_free 1
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@63]  name [Vacuum] ab [Va]
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@75]  name [Vacuum] ab [Va] is_now_free 1
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@63]  name [VacuumT] ab [VT]
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@75]  name [VacuumT] ab [VT] is_now_free 1
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@63]  name [photocathode] ab [ph]
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@75]  name [photocathode] ab [ph] is_now_free 1
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@63]  name [photocathode_3inch] ab [ph]
    2020-06-02 22:58:37.661 INFO  [20290] [SAbbrev::init@75]  name [photocathode_3inch] ab [ph] is_now_free 0
    python: /home/blyth/opticks/sysrap/SAbbrev.cc:81: void SAbbrev::init(): Assertion `is_now_free && "failed to abbreviate "' failed.
    Aborted (core dumped)
    [blyth@localhost ~]$ 



