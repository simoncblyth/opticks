export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation
=============================================================


context
---------

* :doc:`shakedown-running-from-binary-dist`


issue
------

Want to move to new geometry workflow where the geocache is 
created entirely from the modern GDML file (geant4 10.4.2) with 
DYB geometry : because it is familiar and useful for fast testing.

Currently with JUNO geometry, the tests CG4Test and OKG4Test 
take 15 mins each compared to 20s(?) for DYB. Clearly this
quick geometry is very convenient for testing.

But DYB geometry .gdml is exported from an old Geant4 that 
lacked many things, which the .dae included.  So need a boatload
of G4DAE + COLLADA reading assimp code to create the geocache 
from these old geometry files.  

That doesnt fit with the modern geometry workflow. So can 
I export the GDML geometry and use that, allowing the legacy 
code to be retired ?


Creating geocache from the first GDML export
-----------------------------------------------

* avoided this problem below by not treating as const properties 


::

    [blyth@localhost ~]$ geocache-dx-v0
    === o-cmdline-binary-match : --okx4
    === o-gdb-update : placeholder
    === o-main : /home/blyth/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml -runfolder geocache-dx-v0 --runcomment export-dyb-near-for-regeneration ======= PWD /tmp/blyth/opticks/geocache-create- Tue Sep 17 21:55:08 CST 2019
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 0 /home/blyth/local/opticks/lib/OKX4Test
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 1 --okx4
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 2 --g4codegen
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 3 --deletegeocache
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 4 --gdmlpath
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 5 /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 6 -runfolder
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 7 geocache-dx-v0
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 8 --runcomment
    2019-09-17 21:55:08.928 INFO  [130758] [main@90] 9 export-dyb-near-for-regeneration
    2019-09-17 21:55:08.928 INFO  [130758] [main@107]  csgskiplv NONE
    2019-09-17 21:55:08.928 INFO  [130758] [main@111]  parsing /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml'...
    G4GDML: Reading definitions...

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : InvalidExpression
          issued by : G4GDMLEvaluator::DefineConstant()
    Redefinition of constant or variable: SCINTILLATIONYIELD
    *** Fatal Exception ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------


Indeed the below are duplicated::

   71     <constant name="SCINTILLATIONYIELD" value="10"/>
   72     <constant name="RESOLUTIONSCALE" value="1"/>
   73     <constant name="FASTTIMECONSTANT" value="3.6399998664856"/>
   74     <constant name="SLOWTIMECONSTANT" value="12.1999998092651"/>
   75     <constant name="YIELDRATIO" value="0.860000014305115"/>



Comment out and try again::

    opticksdata-dx-vi
    geocache-dx-v0

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : ReadError
          issued by : G4GDMLReadDefine::getMatrix()
    Matrix 'SCINTILLATIONYIELD' was not found!
    *** Fatal Exception ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------



Examining GDML and DAE
-----------------------

* Looks like some DAE to G4 conversion is squashing matrix unto a constant prop when 
  that is not what g4 expects on subsequent parse of the GDML


opticksdata-jv5-vi JUNO GDML matrix is present even for constants::

   012     <matrix coldim="2" name="SCINTILLATIONYIELD0x4b73870" values="-1 11522 1 11522"/>
    13     <matrix coldim="2" name="RESOLUTIONSCALE0x4b73950" values="-1 1 1 1"/>
    14     <matrix coldim="2" name="GammaFASTTIMECONSTANT0x4b73a70" values="-1 4.93 1 4.93"/>
   ...
   139     <material name="LS0x4b61c70" state="solid">
   140       <property name="RINDEX" ref="RINDEX0x4b6e620"/>
   141       <property name="GROUPVEL" ref="GROUPVEL0x4b6ecd0"/>
   142       <property name="RAYLEIGH" ref="RAYLEIGH0x4b736f0"/>
   143       <property name="ABSLENGTH" ref="ABSLENGTH0x4b6ee30"/>
   144       <property name="FASTCOMPONENT" ref="FASTCOMPONENT0x4b710b0"/>
   145       <property name="SLOWCOMPONENT" ref="SLOWCOMPONENT0x4b72290"/>
   146       <property name="REEMISSIONPROB" ref="REEMISSIONPROB0x4b73470"/>
   147       <property name="SCINTILLATIONYIELD" ref="SCINTILLATIONYIELD0x4b73870"/>
   148       <property name="RESOLUTIONSCALE" ref="RESOLUTIONSCALE0x4b73950"/>


opticksdata-d0-dae-vi::

     65864         <matrix coldim="2" name="FASTTIMECONSTANT0xc410e40">-1 3.64 1 3.64</matrix>
     65865         <property name="FASTTIMECONSTANT" ref="FASTTIMECONSTANT0xc410e40"/>
     ...
     65891         <property name="ReemissionYIELDRATIO" ref="ReemissionYIELDRATIO0xc411798"/>
     65892         <matrix coldim="2" name="SCINTILLATIONYIELD0xc4107b8">-1 11522 1 11522</matrix>
     65893         <property name="SCINTILLATIONYIELD" ref="SCINTILLATIONYIELD0xc4107b8"/>
     65894         <matrix coldim="2" name="SLOWCOMPONENT0xc402770">1.55e-06 0   .. great big matrix elided ...
     65895         <property name="SLOWCOMPONENT" ref="SLOWCOMPONENT0xc402770"/>
     65896         <matrix coldim="2" name="SLOWTIMECONSTANT0xc411120">-1 12.2 1 12.2</matrix>
     65897         <property name="SLOWTIMECONSTANT" ref="SLOWTIMECONSTANT0xc411120"/>
     65898         <matrix coldim="2" name="YIELDRATIO0xc411170">-1 0.86 1 0.86</matrix>
     65899         <property name="YIELDRATIO" ref="YIELDRATIO0xc411170"/>
     65900       </extra>
     65901     </material>



     65902     <material id="__dd__Materials__Bialkali0xc2f2428">
     65903       <instance_effect url="#__dd__Materials__Bialkali_fx_0xc2f2428"/>
     65904       <extra>
     65905         <matrix coldim="2" name="ABSLENGTH0xc0b7a90">1.55e-06 0.0001 1.61e-06 500 2.07e-06 1000 2.48e-06 2000 3.56e-06 1000 4.13e-06 1000 6.2e-06 1000 1.033e-05 1000 1.55e-05 1000</matrix>
     65906         <property name="ABSLENGTH" ref="ABSLENGTH0xc0b7a90"/>
     65907         <matrix coldim="2" name="EFFICIENCY0xc2c6598">1.55e-06 0.0001 1.8e-06 0.002 1.9e-06 0.005 2e-06 0.01 2.05e-06 0.017 2.16e-06 0.03 2.19e-06 0.04 2.23e-06 0.05 2.27e-06 0.06 2.32e-06 0.07 2.36e-06 0.08 2.41e-06 0.09 2.46e-06 0.1 2.5e-06 0.11 2.56e-06 0.13 2.6       1e-06 0.15 2.67e-06 0.16 2.72e-06 0.18 2.79e-06 0.19 2.85e-06 0.2 2.92e-06 0.21 2.99e-06 0.22 3.06e-06 0.22 3.14e-06 0.23 3.22e-06 0.24 3.31e-06 0.24 3.4e-06 0.24 3.49e-06 0.23 3.59e-06 0.22 3.7e-06 0.21 3.81e-06 0.17 3.94e-06 0.14 4.07e-06 0.09 4.1e-06 0.035 4.4e-       06 0.005 5e-06 0.001 6.2e-06 0.0001 1.033e-05 0 1.55e-05 0</matrix>
     65908         <property name="EFFICIENCY" ref="EFFICIENCY0xc2c6598"/>
     65909         <matrix coldim="2" name="RINDEX0xc0fd260">1.55e-06 1.458 2.07e-06 1.458 4.13e-06 1.458 6.2e-06 1.458 1.033e-05 1.458 1.55e-05 1.458</matrix>
     65910         <property name="RINDEX" ref="RINDEX0xc0fd260"/>
     65911       </extra>
     65912     </material>



opticksdata-dx-vi DYB export has duplicated props and hexless refs::

      049     <constant name="SCINTILLATIONYIELD" value="10"/>
       50     <constant name="RESOLUTIONSCALE" value="1"/>
       51     <constant name="FASTTIMECONSTANT" value="3.6399998664856"/>
       52     <constant name="SLOWTIMECONSTANT" value="12.1999998092651"/>
       53     <constant name="YIELDRATIO" value="0.860000014305115"/>
      ...  
       72 <!--
       73     <constant name="SCINTILLATIONYIELD" value="10"/>
       74     <constant name="RESOLUTIONSCALE" value="1"/>
       75     <constant name="FASTTIMECONSTANT" value="3.6399998664856"/>
       76     <constant name="SLOWTIMECONSTANT" value="12.1999998092651"/>
       77     <constant name="YIELDRATIO" value="0.860000014305115"/>
       78 -->


      593     <material name="/dd/Materials/LiquidScintillator0x442d3c0" state="solid">
      594       <property name="RINDEX" ref="RINDEX0x234c610"/>
      595       <property name="GROUPVEL" ref="GROUPVEL0x234db00"/>
      596       <property name="RAYLEIGH" ref="RAYLEIGH0x234d4c0"/>
      597       <property name="ABSLENGTH" ref="ABSLENGTH0x234d1a0"/>
      598       <property name="FASTCOMPONENT" ref="FASTCOMPONENT0x234d060"/>
      599       <property name="SLOWCOMPONENT" ref="SLOWCOMPONENT0x234e1e0"/>
      600       <property name="REEMISSIONPROB" ref="REEMISSIONPROB0x234d7e0"/>
      601       <property name="SCINTILLATIONYIELD" ref="SCINTILLATIONYIELD"/>
      602       <property name="RESOLUTIONSCALE" ref="RESOLUTIONSCALE"/>
      603       <property name="FASTTIMECONSTANT" ref="FASTTIMECONSTANT"/>
      604       <property name="SLOWTIMECONSTANT" ref="SLOWTIMECONSTANT"/>
      605       <property name="YIELDRATIO" ref="YIELDRATIO"/>




Convertion of GGeo GMaterial into Geant4 props
--------------------------------------------------

Recall:

*  x4 : Geant4 -> GGeo
* cfg4 : GGeo -> Geant4 


::

    236 G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial* ggmat)
    237 {
    ...
    292     if(is_scintillator)
    293     {
    294         GPropertyMap<float>* scintillator = m_sclib->getRaw(name);
    295         assert(scintillator && "non-zero reemission prob materials should has an associated raw scintillator");
    296         LOG(debug) << "CPropLib::makeMaterialPropertiesTable found corresponding scintillator from sclib "
    297                   << " name " << name
    298                   << " keys " << scintillator->getKeysString()
    299                    ;
    300         bool keylocal, constant ;
    301         addProperties(mpt, scintillator, "SLOWCOMPONENT,FASTCOMPONENT", keylocal=false, constant=false);
    302         addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal=false, constant=true );
    303 
    304         // NB the above skips prefixed versions of the constants: Alpha, 
    305         //addProperties(mpt, scintillator, "ALL",          keylocal=false, constant=true );
    306     }
    307     return mpt ;
    308 }



::

    151 CGDMLDetector::addMPTLegacyGDML
    152 -----------------------------------
    153 
    154 The GDML exported by geant4 that comes with nuwa lack material properties 
    155 so use the properties from the G4DAE export, to enable recovery of the materials.




Look for property overrides
----------------------------------------------

::

    [blyth@localhost issues]$ opticks-f SCINTILLATIONYIELD
    ./cfg4/CPropLib.cc:    gdls["SCINTILLATIONYIELD"] = yield ;  
    ./cfg4/CPropLib.cc:    ls["SCINTILLATIONYIELD"] = yield ;  
    ./cfg4/CPropLib.cc:        addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal=false, constant=true );
    ./cfg4/cfg4.bash:    2016-06-29 14:53:38.821 WARN  [13144929] [CPropLib::addConstProperty@401] CPropLib::addConstProperty OVERRIDE GdDopedLS.SCINTILLATIONYIELD from 11522 to 10
    ./cfg4/cfg4.bash:    2016-06-29 14:53:38.821 WARN  [13144929] [CPropLib::addConstProperty@401] CPropLib::addConstProperty OVERRIDE LiquidScintillator.SCINTILLATIONYIELD from 11522 to 10
    ./cfg4/Scintillation.cc:                                      GetConstProperty("SCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("PROTONSCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("DEUTERONSCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("TRITONSCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("ALPHASCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("IONSCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("ELECTRONSCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("ELECTRONSCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:      GetProperty("ELECTRONSCINTILLATIONYIELD");
    ./cfg4/Scintillation.cc:       << "ELECTRONSCINTILLATIONYIELD is set by the user\n"
    ./cfg4/DsG4Scintillation.cc:            const G4MaterialPropertyVector* ptable = aMaterialPropertiesTable->GetProperty("SCINTILLATIONYIELD");
    ./cfg4/DsG4Scintillation.cc:                ScintillationYield = aMaterialPropertiesTable->GetConstProperty("SCINTILLATIONYIELD") ;
    ./cfg4/DsG4Scintillation.cc:                LOG(fatal) << "Failed to get SCINTILLATIONYIELD" ;
    ./examples/Geant4/OpNovice/src/OpNoviceDetectorConstruction.cc:  myMPT1->AddConstProperty("SCINTILLATIONYIELD",50./MeV);
    ./extg4/LXe_Materials.cc:  fLXe_mt->AddConstProperty("SCINTILLATIONYIELD",12000./MeV);
    ./extg4/LXe_Materials.cc:  fMPTPStyrene->AddConstProperty("SCINTILLATIONYIELD",10./keV);
    ./extg4/X4OpNoviceMaterials.cc:  myMPT1->AddConstProperty("SCINTILLATIONYIELD",50./MeV);
    ./extg4/OpNoviceDetectorConstruction.cc:  myMPT1->AddConstProperty("SCINTILLATIONYIELD",50./MeV);
    [blyth@localhost opticks]$ 


::

    119 void CPropLib::initSetupOverrides()
    120 {
    121     float yield = 10.f ;
    122 
    123     std::map<std::string, float>  gdls ;
    124     gdls["SCINTILLATIONYIELD"] = yield ;
    125 
    126     std::map<std::string, float>  ls ;
    127     ls["SCINTILLATIONYIELD"] = yield ;
    128 
    129     m_const_override["GdDopedLS"] = gdls ;
    130     m_const_override["LiquidScintillator"] = ls ;
    131 }




Exporting DYB geometry as GDML ?
-------------------------------------------

How to export DYB into GDML ?::

    CGeometry::export_

There are options::

   --export
   --exportconfig $TMP   # the default   getExportConfig


::

    148 void CGeometry::export_()
    149 {
    150     bool expo = m_cfg->hasOpt("export");
    151     if(!expo) return ;
    152     //std::string expodir = m_cfg->getExportConfig();
    153 
    154     const char* expodir = "$TMP/CGeometry" ;
    155 
    156     if(BFile::ExistsDir(expodir))
    157     {
    158         BFile::RemoveDir(expodir);
    159         LOG(info) << "CGeometry::export_ removed " << expodir ;
    160     }
    161 
    162     BFile::CreateDir(expodir);
    163     m_detector->export_dae(expodir, "CGeometry.dae");
    164     m_detector->export_gdml(expodir, "CGeometry.gdml");
    165 }


::

    OKG4Test --export 
    ...
    2019-09-17 21:32:31.254 INFO  [96888] [CGDML::Export@65] export to /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml

::

    cd /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
    cp /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml g4_00_CGeometry_export.gdml


::

    [blyth@localhost cfg4]$ opticksdata-d         ## original old GDML with lots of things missing 
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml

                                                 ## original dae that carried the missing properties  
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae

                                       
    [blyth@localhost ~]$ opticksdata-dx          ## recent "--export" which uses geant4 10.4.2  
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export.gdml



Follow the geometry loading and export::

   CGDMLDetector=ERROR CPropLib=ERROR OKG4Test --export 

Yep tis all happening on load::

    2019-09-18 21:51:53.877 ERROR [83867] [CPropLib::init@90] [
    2019-09-18 21:51:53.877 ERROR [83867] [CPropLib::init@92] GSurfaceLib numSurfaces 48 this 0x2211b20 basis 0 isClosed 1 hasDomain 1
    2019-09-18 21:51:53.877 ERROR [83867] [CPropLib::init@115] ]
    2019-09-18 21:51:53.877 ERROR [83867] [CGDMLDetector::CGDMLDetector@62] [
    2019-09-18 21:51:53.877 ERROR [83867] [CGDMLDetector::init@89] parse /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml' done!
    2019-09-18 21:51:54.199 FATAL [83867] [CMaterialSort::sort@83]  sorting G4MaterialTable using order kv 38
    2019-09-18 21:51:54.409 ERROR [83867] [CGDMLDetector::addMPTLegacyGDML@168]  nmat 36 nmat_without_mpt 36
    2019-09-18 21:51:54.409 ERROR [83867] [CGDMLDetector::addMPTLegacyGDML@186]  ALL G4 MATERIALS LACK MPT  FIXING USING Opticks MATERIALS 
    2019-09-18 21:51:54.411 ERROR [83867] [CPropLib::addConstProperty@404]  OVERRIDE GdDopedLS.SCINTILLATIONYIELD from 11522 to 10
    2019-09-18 21:51:54.411 ERROR [83867] [CPropLib::addConstProperty@404]  OVERRIDE LiquidScintillator.SCINTILLATIONYIELD from 11522 to 10
    2019-09-18 21:51:54.412 ERROR [83867] [CPropLib::makeMaterialPropertiesTable@275]  name Bialkali adding EFFICIENCY : START GPropertyMap  type skinsurface name /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2019-09-18 21:51:54.415 INFO  [83867] [CGDMLDetector::addMPTLegacyGDML@224] CGDMLDetector::addMPT added MPT to 36 g4 materials 
    2019-09-18 21:51:54.415 ERROR [83867] [CGDMLDetector::init@102]  skip standardizeGeant4MaterialProperties in legacy running 



Try not treating as constant properties::

    308 void CPropLib::addScintillatorMaterialProperties( G4MaterialPropertiesTable* mpt, const char* name )
    309 {   
    310     GPropertyMap<float>* scintillator = m_sclib->getRaw(name);
    311     assert(scintillator && "non-zero reemission prob materials should has an associated raw scintillator");
    312     LOG(LEVEL) 
    313         << " found corresponding scintillator from sclib "
    314         << " name " << name 
    315         << " keys " << scintillator->getKeysString() 
    316         ; 
    317        
    318     bool keylocal = false ;
    319     bool constant = false ;
    320     addProperties(mpt, scintillator, "SLOWCOMPONENT,FASTCOMPONENT", keylocal, constant);
    321     addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal, constant ); // this used constant=true formerly
    322 
    323     // NB the above skips prefixed versions of the constants: Alpha, 
    324     //addProperties(mpt, scintillator, "ALL",          keylocal=false, constant=true );
    325 }


Exports to::

   /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml

    182 opticksdata-dxtmp(){    echo $(opticks-dir)/tmp/CGeometry/CGeometry.gdml ; }
    183 opticksdata-dxtmp-vi(){ vi $(opticksdata-dxtmp) ; }

    384 geocache-dxtmp-(){  opticksdata- ; geocache-create- --gdmlpath $(opticksdata-dxtmp) $* ; }
    385 geocache-dxtmp-comment(){  echo gdml-insitu-created-by-OKG4Test-export ; }
    386 geocache-dxtmp(){   $FUNCNAME- -runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }
     



Not treating as constant yields a repetitious matrix::

    72     <matrix coldim="2" name="RESOLUTIONSCALE0x39c9060" values="1.512e-06 1 1.5498e-06 1 1.58954e-06 1 1.63137e-06 1 1.67546e-06 1 1.722e-06 1 1.7712e-06 1 1.8233e-06 1 1.87855e-06 1 1.93725e-06 1 1.99974e-06 1 2.0664e-06 1 2.13766e-06 1 2.214e-06 1 2.296e-06 1 2.384      31e-06 1 2.47968e-06 1 2.583e-06 1 2.69531e-06 1 2.81782e-06 1 2.952e-06 1 3.0996e-06 1 3.26274e-06 1 3.44401e-06 1 3.64659e-06 1 3.87451e-06 1 4.13281e-06 1 4.42801e-06 1 4.76862e-06 1 5.16601e-06 1 5.63564e-06 1 6.19921e-06 1 6.88801e-06 1 7.74901e-06 1 8.85601e-0      6 1 1.0332e-05 1 1.23984e-05 1 1.5498e-05 1 2.0664e-05 1"/>
       73     <matrix coldim="2" name="SCINTILLATIONYIELD0x39c6f50" values="1.512e-06 11522 1.5498e-06 11522 1.58954e-06 11522 1.63137e-06 11522 1.67546e-06 11522 1.722e-06 11522 1.7712e-06 11522 1.8233e-06 11522 1.87855e-06 11522 1.93725e-06 11522 1.99974e-06 11522 2.0664e-0      6 11522 2.13766e-06 11522 2.214e-06 11522 2.296e-06 11522 2.38431e-06 11522 2.47968e-06 11522 2.583e-06 11522 2.69531e-06 11522 2.81782e-06 11522 2.952e-06 11522 3.0996e-06 11522 3.26274e-06 11522 3.44401e-06 11522 3.64659e-06 11522 3.87451e-06 11522 4.13281e-06 115      22 4.42801e-06 11522 4.76862e-06 11522 5.16601e-06 11522 5.63564e-06 11522 6.19921e-06 11522 6.88801e-06 11522 7.74901e-06 11522 8.85601e-06 11522 1.0332e-05 11522 1.23984e-05 11522 1.5498e-05 11522 2.0664e-05 11522"/>
       74     <matrix coldim="2" name="SLOWTIMECONSTANT0x39c98b0" values="1.512e-06 12.2 1.5498e-06 12.2 1.58954e-06 12.2 1.63137e-06 12.2 1.67546e-06 12.2 1.722e-06 12.2 1.7712e-06 12.2 1.8233e-06 12.2 1.87855e-06 12.2 1.93725e-06 12.2 1.99974e-06 12.2 2.0664e-06 12.2 2.1376      6e-06 12.2 2.214e-06 12.2 2.296e-06 12.2 2.38431e-06 12.2 2.47968e-06 12.2 2.583e-06 12.2 2.69531e-06 12.2 2.81782e-06 12.2 2.952e-06 12.2 3.0996e-06 12.2 3.26274e-06 12.2 3.44401e-06 12.2 3.64659e-06 12.2 3.87451e-06 12.2 4.13281e-06 12.2 4.42801e-06 12.2 4.76862e-      06 12.2 5.16601e-06 12.2 5.63564e-06 12.2 6.19921e-06 12.2 6.88801e-06 12.2 7.74901e-06 12.2 8.85601e-06 12.2 1.0332e-05 12.2 1.23984e-05 12.2 1.5498e-05 12.2 2.0664e-05 12.2"/>







This non-const GDML does better
----------------------------------

::

    [blyth@localhost cfg4]$ geocache-
    [blyth@localhost cfg4]$ geocache-dxtmp
    === o-cmdline-binary-match : --okx4
    === o-gdb-update : placeholder
    === o-main : /home/blyth/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml -runfolder geocache-dxtmp --runcomment gdml-insitu-created-by-OKG4Test-export ======= PWD /tmp/blyth/opticks/geocache-create- Wed Sep 18 22:22:49 CST 2019
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 0 /home/blyth/local/opticks/lib/OKX4Test
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 1 --okx4
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 2 --g4codegen
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 3 --deletegeocache
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 4 --gdmlpath
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 5 /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 6 -runfolder
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 7 geocache-dxtmp
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 8 --runcomment
    2019-09-18 22:22:49.490 INFO  [131088] [main@90] 9 gdml-insitu-created-by-OKG4Test-export
    2019-09-18 22:22:49.490 INFO  [131088] [main@107]  csgskiplv NONE
    2019-09-18 22:22:49.490 INFO  [131088] [main@111]  parsing /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml
    G4GDML: Reading '/home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml' done!
    2019-09-18 22:22:49.896 INFO  [131088] [main@114] ///////////////////////////////// 
    2019-09-18 22:22:49.928 ERROR [131088] [main@122]  SetKey OKX4Test.X4PhysicalVolume.World0xc15cfc00x5d42890_PV.5aa828335373870398bf4f738781da6c
    2019-09-18 22:22:49.932 INFO  [131088] [Opticks::init@389] INTEROP_MODE
    2019-09-18 22:22:49.932 FATAL [131088] [Opticks::init@392] OPTICKS_LEGACY_GEOMETRY_ENABLED mode is active  : ie dae src access to geometry, opticksdata  
    the argument ('unfolder') for option '--recordmax' is invalid
    2019-09-18 22:22:49.934 INFO  [131088] [Opticks::configure@2117]  setting CUDA_VISIBLE_DEVICES envvar internally to 1

    ...
    2019-09-18 22:23:16.335 ERROR [131088] [nnode::to_g4code_r@804] no g4code on left/right :  prim in G4, but tree in Opticks perhaps ? 
    2019-09-18 22:23:16.339 INFO  [131088] [X4PhysicalVolume::convertSolid@666] ] 210
    2019-09-18 22:23:16.339 INFO  [131088] [X4PhysicalVolume::convertSolid@634] [ 211 soname near_pool_iws_box0xc288ce80x5b001a0 lvname /dd/Geometry/Pool/lvNearPoolIWS0xc28bc600x5b473e0
    2019-09-18 22:23:16.341 INFO  [131088] [NTreeBalance<T>::create_balanced@59] op_mask intersection 
    2019-09-18 22:23:16.342 INFO  [131088] [NTreeBalance<T>::create_balanced@60] hop_mask intersection 
    2019-09-18 22:23:16.342 INFO  [131088] [NTreeBalance<T>::create_balanced@73]  CommonTree prims 13
    2019-09-18 22:23:16.395 ERROR [131088] [X4CSG::generateTestMain@255]  skip as no g4code 
    2019-09-18 22:23:16.395 INFO  [131088] [NTreeBalance<T>::create_balanced@59] op_mask intersection 
    2019-09-18 22:23:16.395 INFO  [131088] [NTreeBalance<T>::create_balanced@60] hop_mask intersection 
    2019-09-18 22:23:16.395 INFO  [131088] [NTreeBalance<T>::create_balanced@73]  CommonTree prims 13



But some solid (lvIDx 211) has polgonization problem::

    (gdb) bt
    #0  0x00007ff8fd2397c2 in HepGeom::Plane3D<double>::distance (this=0x7fffc673e1d0, p=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/include/CLHEP/Geometry/Plane3D.h:103
    #1  0x00007ff8fcae066c in BooleanProcessor::testFaceVsPlane (this=0x7fffc673e690, edge=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/BooleanProcessor.src:669
    #2  0x00007ff8fcae1a08 in BooleanProcessor::testFaceVsFace (this=0x7fffc673e690, iface1=5, iface2=89) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/BooleanProcessor.src:986
    #3  0x00007ff8fcae622e in BooleanProcessor::execute (this=0x7fffc673e690, op=2, a=..., b=..., err=@0x7fffc673e7ac: 0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/BooleanProcessor.src:2131
    #4  0x00007ff8fcae6d62 in HepPolyhedronProcessor::execute1 (this=0x7fffc673ec00, a_poly=..., a_is=std::vector of length 12, capacity 12 = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:171
    #5  0x00007ff8fcae85cb in HepPolyhedron_exec::visit (this=0x7fffc673eb90, a_is=std::vector of length 12, capacity 12 = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:131
    #6  0x00007ff8fcae84b5 in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=11, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:94
    #7  0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=10, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #8  0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=9, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #9  0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=8, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #10 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=7, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #11 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=6, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #12 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=5, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #13 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=4, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #14 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=3, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #15 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=2, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #16 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=1, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #17 0x00007ff8fcae84db in HEPVis::bijection_visitor::visit (this=0x7fffc673eb90, a_level=0, a_is=std::list = {...}) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:96
    #18 0x00007ff8fcae83ef in HEPVis::bijection_visitor::visitx (this=0x7fffc673eb90) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:84
    #19 0x00007ff8fcae6c13 in HepPolyhedronProcessor::execute (this=0x7fffc673ec00, a_poly=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/graphics_reps/src/HepPolyhedronProcessor.src:147
    #20 0x00007ff8fd2b8483 in G4SubtractionSolid::CreatePolyhedron (this=0x2c207d0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc:591
    #21 0x00007ff9041f54fd in X4Mesh::polygonize (this=0x7fffc673f6c0) at /home/blyth/opticks/extg4/X4Mesh.cc:154
    #22 0x00007ff9041f505b in X4Mesh::init (this=0x7fffc673f6c0) at /home/blyth/opticks/extg4/X4Mesh.cc:119
    #23 0x00007ff9041f502e in X4Mesh::X4Mesh (this=0x7fffc673f6c0, solid=0x2c207d0) at /home/blyth/opticks/extg4/X4Mesh.cc:109
    #24 0x00007ff9041f4f9b in X4Mesh::Convert (solid=0x2c207d0) at /home/blyth/opticks/extg4/X4Mesh.cc:93
    #25 0x00007ff904206b48 in X4PhysicalVolume::convertSolid (this=0x7fffc67418c0, lvIdx=211, soIdx=211, solid=0x2c207d0, lvname="/dd/Geometry/Pool/lvNearPoolIWS0xc28bc600x5b473e0", balance_deep_tree=true) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:663
    #26 0x00007ff90420642f in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2d3b780, depth=7) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:566
    #27 0x00007ff9042062ce in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2d3dc70, depth=6) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:551
    #28 0x00007ff9042062ce in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2e946d0, depth=5) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:551
    #29 0x00007ff9042062ce in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2e95530, depth=4) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:551
    #30 0x00007ff9042062ce in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2e96c60, depth=3) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:551
    #31 0x00007ff9042062ce in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2e97bf0, depth=2) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:551
    #32 0x00007ff9042062ce in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2e97d50, depth=1) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:551
    #33 0x00007ff9042062ce in X4PhysicalVolume::convertSolids_r (this=0x7fffc67418c0, pv=0x2154470, depth=0) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:551
    #34 0x00007ff904206073 in X4PhysicalVolume::convertSolids (this=0x7fffc67418c0) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:509
    #35 0x00007ff9042049fb in X4PhysicalVolume::init (this=0x7fffc67418c0) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:186
    #36 0x00007ff904204820 in X4PhysicalVolume::X4PhysicalVolume (this=0x7fffc67418c0, ggeo=0x2137450, top=0x2154470) at /home/blyth/opticks/extg4/X4PhysicalVolume.cc:170
    #37 0x000000000040523e in main (argc=10, argv=0x7fffc6742728) at /home/blyth/opticks/okg4/tests/OKX4Test.cc:144
    (gdb) 


::

    [blyth@localhost opticks]$ geocache-dxtmp-keydir
    /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x5d42890_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1

    [blyth@localhost opticks]$ cd /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x5d42890_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    [blyth@localhost 1]$ l
    total 0
    drwxrwxr-x. 3 blyth blyth 19 Sep 18 22:22 g4codegen
    [blyth@localhost 1]$ cd g4codegen/
    [blyth@localhost g4codegen]$ l
    total 16
    drwxrwxr-x. 2 blyth blyth 12288 Sep 18 22:23 tests
    [blyth@localhost g4codegen]$ cd tests/
    [blyth@localhost tests]$ l
    total 1644
    -rw-rw-r--. 1 blyth blyth    0 Sep 18 22:23 x211.cc
    -rw-rw-r--. 1 blyth blyth 7026 Sep 18 22:23 x211.gdml
    -rw-rw-r--. 1 blyth blyth 1709 Sep 18 22:23 x210.cc
    -rw-rw-r--. 1 blyth blyth  344 Sep 18 22:23 x210.gdml
    -rw-rw-r--. 1 blyth blyth 1707 Sep 18 22:23 x209.cc


x211.gdml tis a mega subtraction solid::

     01 <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
      2 <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">
      3 
      4   <solids>
      5     <box lunit="mm" name="near_pool_iws0xc2cab980x5afdaf0" x="13824" y="7824" z="8908"/>
      6     <box lunit="mm" name="near_pool_iws_sub00xc2cabd80x5afdf60" x="3347.67401109936" y="3347.67401109936" z="8918"/>
      7     <subtraction name="near_pool_iws-ChildFornear_pool_iws_box0xc2865680x5afe130">
      8       <first ref="near_pool_iws0xc2cab980x5afdaf0"/>
      9       <second ref="near_pool_iws_sub00xc2cabd80x5afdf60"/>
     10       <position name="near_pool_iws-ChildFornear_pool_iws_box0xc2865680x5afe130_pos" unit="mm" x="6912" y="3912" z="0"/>
     11       <rotation name="near_pool_iws-ChildFornear_pool_iws_box0xc2865680x5afe130_rot" unit="deg" x="0" y="0" z="45"/>
     12     </subtraction>
     13     <box lunit="mm" name="near_pool_iws_sub10xc2cac180x5afe0d0" x="3347.67401109936" y="3347.67401109936" z="8918"/>
     14     <subtraction name="near_pool_iws-ChildFornear_pool_iws_box0xc2866d00x5afe440">
     15       <first ref="near_pool_iws-ChildFornear_pool_iws_box0xc2865680x5afe130"/>
     16       <second ref="near_pool_iws_sub10xc2cac180x5afe0d0"/>
     17       <position name="near_pool_iws-ChildFornear_pool_iws_box0xc2866d00x5afe440_pos" unit="mm" x="6912" y="-3912" z=


    [blyth@localhost tests]$ cat x232.gdml
    <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">

      <solids>
        <box lunit="mm" name="near_pool_ows0xc2bc1d80x5b053f0" x="15832" y="9832" z="9912"/>
        <box lunit="mm" name="near_pool_ows_sub00xc55ebf80x5b05870" x="4179.41484434453" y="4179.41484434453" z="9922"/>
        <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c1480x5b05a40">
          <first ref="near_pool_ows0xc2bc1d80x5b053f0"/>
          <second ref="near_pool_ows_sub00xc55ebf80x5b05870"/>
          <position name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c1480x5b05a40_pos" unit="mm" x="7916" y="4916" z="0"/>
          <rotation name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c1480x5b05a40_rot" unit="deg" x="0" y="0" z="45"/>
        </subtraction>
        <box lunit="mm" name="near_pool_ows_sub10xc21e9400x5b059e0" x="4179.41484434453" y="4179.41484434453" z="9922"/>
        <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xc12f6400x5b05d50">
          <first ref="near_pool_ows-ChildFornear_pool_ows_box0xbf8c1480x5b05a40"/>
          <second ref="near_pool_ows_sub10xc21e9400x5b059e0"/>




Succeeds to create a geocache from the non-const GDML but the viz setup is way off.  
No geo selection, means that very old huge world problem again.

::

    geocache-dxtmp --x4polyskip 211,232

    ...


    2019-09-18 22:50:20.165 INFO  [174606] [OpticksProfile::dump@401]  npy 69,4 /home/blyth/local/opticks/tmp/source/evt/g4live/torch/OpticksProfile.npy
    2019-09-18 22:50:20.165 INFO  [174606] [OpticksProfile::accumulateDump@287] Opticks::postgeocache nacc 0
    2019-09-18 22:50:20.165 INFO  [174606] [Opticks::reportGeoCacheCoordinates@920]  ok.idpath  /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x5d42890_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    2019-09-18 22:50:20.165 INFO  [174606] [Opticks::reportGeoCacheCoordinates@921]  ok.keyspec OKX4Test.X4PhysicalVolume.World0xc15cfc00x5d42890_PV.5aa828335373870398bf4f738781da6c
    2019-09-18 22:50:20.166 INFO  [174606] [Opticks::reportGeoCacheCoordinates@922]  To reuse this geometry: 
    2019-09-18 22:50:20.166 INFO  [174606] [Opticks::reportGeoCacheCoordinates@923]    1. set envvar OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x5d42890_PV.5aa828335373870398bf4f738781da6c
    2019-09-18 22:50:20.166 INFO  [174606] [Opticks::reportGeoCacheCoordinates@924]    2. enable envvar sensitivity with --envkey argument to Opticks executables 
    2019-09-18 22:50:20.166 FATAL [174606] [Opticks::reportGeoCacheCoordinates@932] THE LIVE keyspec DOES NOT MATCH THAT OF THE CURRENT ENVVAR 
    2019-09-18 22:50:20.166 INFO  [174606] [Opticks::reportGeoCacheCoordinates@933]  (envvar) OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    2019-09-18 22:50:20.166 INFO  [174606] [Opticks::reportGeoCacheCoordinates@934]  (live)   OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x5d42890_PV.5aa828335373870398bf4f738781da6c
    2019-09-18 22:50:20.166 INFO  [174606] [Opticks::dumpRC@227]  rc 0 rcmsg : -
    === o-main : /home/blyth/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml --runfolder geocache-dxtmp --runcomment gdml-insitu-created-by-OKG4Test-export --x4polyskip 211,232 ======= PWD /tmp/blyth/opticks/geocache-create- RC 0 Wed Sep 18 22:50:20 CST 2019
    echo o-postline : dummy
    o-postline : dummy
    /home/blyth/local/opticks/bin/o.sh : RC : 0
    [blyth@localhost ~]$ 
    [blyth@localhost ~]$ 




OKTest : non-legacy with the newly created DYB geocache
-----------------------------------------------------------

vip::

    146 export OPTICKS_KEY_JV5=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce   ## geocache-j1808-v5-export 
    147 export OPTICKS_KEY_DYBXTMP=OKX4Test.X4PhysicalVolume.World0xc15cfc00x5d42890_PV.5aa828335373870398bf4f738781da6c
    148 export OPTICKS_KEY=$OPTICKS_KEY_DYBXTMP
    149 unset OPTICKS_LEGACY_GEOMETRY_ENABLED
    150 #export OPTICKS_LEGACY_GEOMETRY_ENABLED=1


::
    
    OKTest

    /home/blyth/opticks/ggeo/GMeshLib.cc:181: void GMeshLib::loadAltReferences(): Assertion `unsigned(altindex) < m_meshes.size()' failed.
    
    Program received signal SIGABRT, Aborted.
    (gdb) bt
    ...
    #4  0x00007ffff50e5c8b in GMeshLib::loadAltReferences (this=0x1a1ee00) at /home/blyth/opticks/ggeo/GMeshLib.cc:181
    #5  0x00007ffff50e5614 in GMeshLib::loadFromCache (this=0x1a1ee00) at /home/blyth/opticks/ggeo/GMeshLib.cc:71
    #6  0x00007ffff50e555d in GMeshLib::Load (ok=0x626840) at /home/blyth/opticks/ggeo/GMeshLib.cc:59
    #7  0x00007ffff50da828 in GGeo::loadFromCache (this=0x653e40) at /home/blyth/opticks/ggeo/GGeo.cc:898
    #8  0x00007ffff50d8b5f in GGeo::loadGeometry (this=0x653e40) at /home/blyth/opticks/ggeo/GGeo.cc:626
    #9  0x00007ffff64e1cff in OpticksGeometry::loadGeometryBase (this=0x653730) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:156
    #10 0x00007ffff64e1723 in OpticksGeometry::loadGeometry (this=0x653730) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:98
    #11 0x00007ffff64e640a in OpticksHub::loadGeometry (this=0x640bd0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:546
    #12 0x00007ffff64e4e4e in OpticksHub::init (this=0x640bd0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:253
    #13 0x00007ffff64e4b3f in OpticksHub::OpticksHub (this=0x640bd0, ok=0x626840) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:217
    #14 0x00007ffff7bd59cf in OKMgr::OKMgr (this=0x7fffffffd780, argc=1, argv=0x7fffffffd8f8, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:54
    #15 0x0000000000402ead in main (argc=1, argv=0x7fffffffd8f8) at /home/blyth/opticks/ok/tests/OKTest.cc:32
    (gdb) 


::

    (gdb) f 5
    #5  0x00007ffff50e5614 in GMeshLib::loadFromCache (this=0x1a1ee00) at /home/blyth/opticks/ggeo/GMeshLib.cc:71
    71      loadAltReferences();  
    (gdb) list
    66  
    67      m_meshnames = GItemList::Load(idpath, GMESHLIB_LIST, "GItemList" ) ;
    68      assert(m_meshnames);
    69  
    70      loadMeshes(idpath);
    71      loadAltReferences();  
    72  }
    73  
    74  void GMeshLib::save() 
    75  {
    (gdb) p idpath
    $1 = 0x650cb0 "/home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x5d42890_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1"
    (gdb) 


    (gdb) f 4
    #4  0x00007ffff50e5c8b in GMeshLib::loadAltReferences (this=0x1a1ee00) at /home/blyth/opticks/ggeo/GMeshLib.cc:181
    181         assert( unsigned(altindex) < m_meshes.size() ); 
    (gdb) p altindex
    $2 = 250
    (gdb) p m_meshes.size()
    $3 = 250
    (gdb) 



Hmm quite a few altindex out of range ?::

    (gdb) p m_solids[249]->get_altindex()
    $14 = 0
    (gdb) p i
    $15 = 56
    (gdb) p m_solids[56]->get_altindex()
    $16 = 250
    (gdb) p m_solids[55]->get_altindex()
    $17 = -1
    (gdb) p m_solids[57]->get_altindex()
    $18 = 251
    (gdb) p m_solids[58]->get_altindex()
    $19 = 252
    (gdb) p m_solids[59]->get_altindex()
    $20 = 253
    (gdb) p m_solids[60]->get_altindex()
    $21 = 254
    (gdb) p m_solids[61]->get_altindex()
    $22 = 255
    (gdb) p m_solids[62]->get_altindex()
    $23 = 256
    (gdb) p m_solids[63]->get_altindex()
    $24 = -1
    (gdb) p m_solids[64]->get_altindex()
    $25 = -1
    (gdb) p m_solids[65]->get_altindex()
    $26 = 257
    (gdb) p m_solids[66]->get_altindex()
    $27 = -1
    (gdb) p m_solids[67]->get_altindex()
    $28 = -1
    (gdb) p m_solids[68]->get_altindex()
    $29 = -1
    (gdb) p m_solids[69]->get_altindex()
    $30 = 258
    (gdb) p m_solids[70]->get_altindex()
    $31 = -1
    (gdb) p m_solids[75]->get_altindex()
    $36 = -1
    (gdb) 


Looks like MAX_MESH problem ?::

     37 const plog::Severity GMeshLib::LEVEL = debug ;
     38 
     39 const unsigned GMeshLib::MAX_MESH = 250 ;   // <-- hmm 500 too large ? it means a lot of filesystem checking 
     40 
     41



From perusing the geocache looks like 300 would be enough, but may have been effects precache : so up to 300 and export again.

::

    [blyth@localhost issues]$ geocache-
    [blyth@localhost issues]$ geocache-kcd
    /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x5d42890_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    rundate
    20190918_224845
    runstamp
    1568818125
    argline
    /home/blyth/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml --runfolder geocache-dxtmp --runcomment gdml-insitu-created-by-OKG4Test-export --x4polyskip 211,232 
    runcomment
    gdml-insitu-created-by-OKG4Test-export
    runlabel
    R0_cvd_1
    runfolder
    geocache-dxtmp
    [blyth@localhost 1]$ pwd
    /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x5d42890_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    [blyth@localhost 1]$ 




Up to 300, export and create geocache again
-----------------------------------------------


Rebuild with limit upped to 300::

    o
    om--

Switch back to legacy geometry mode (vip)::

    export OPTICKS_LEGACY_GEOMETRY_ENABLED=1 

    ini

    OKTest   # check get the usual DYB geometry via legacy route, resource handling revolves around G4DAE path in legacy 

    OKG4Test --export    # OKG4Test uses CG4 functionality to load the old GDML, merge in info from G4DAE

    ## exports the GDML

::

    [blyth@localhost opticks]$ opticksdata-
    [blyth@localhost opticks]$ opticksdata-dxtmp
    /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml
    [blyth@localhost opticks]$ l $(opticksdata-dxtmp)
    -rw-rw-r--. 1 blyth blyth 4544475 Sep 19 18:37 /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml



     

Now switch to non-legacy (vip) and create geocache::


    150 unset OPTICKS_LEGACY_GEOMETRY_ENABLED 
    151 #export OPTICKS_LEGACY_GEOMETRY_ENABLED=1

    ini

    geocache- ; geocache-dxtmp --x4polyskip 211,232


* there are 0:248 (ie 249) distinct solids but there are altmesh for 23 of those : so total of 249+23=272 
  (if i recall correctly the alt meshes are to simulataneously keep balanced and unbalanced geometry)


::

    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@116]  num_indices_with_alt 23
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 0
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 56
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 57
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 58
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 59
    ...
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 213
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 232
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 234
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 236
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::addAltMeshes@126]  index with alt 245
    2019-09-19 18:42:23.581 INFO  [107750] [GMeshLib::dump@227] addAltMeshes meshnames 272 meshes 272

     i   0 aidx   0 midx   0 name               near_top_cover_box0xc23f9700x42cebb0 mesh  nv     34 nf     64
     i   1 aidx   1 midx   1 name                         RPCStrip0xc04bcb00x42cea80 mesh  nv      8 nf     12
     i   2 aidx   2 midx   2 name                      RPCGasgap140xbf4c6600x42ceb50 mesh  nv      8 nf     12

     i  56 aidx  56 midx  56 name                 RadialShieldUnit0xc3d7da80x42db120 mesh  nv    304 nf    628
     i  57 aidx  57 midx  57 name                    TopESRCutHols0xbf9de100x42e5980 mesh  nv    578 nf   1188

     i  58 aidx  58 midx  58 name                 TopRefGapCutHols0xbf9cef80x42e6860 mesh  nv    296 nf    608
     i  59 aidx  59 midx  59 name                    TopRefCutHols0xbf9bd500x42e7710 mesh  nv    296 nf    608
     i  60 aidx  60 midx  60 name                    BotESRCutHols0xbfa73680x42e8e30 mesh  nv    330 nf    688
     i  61 aidx  61 midx  61 name                 BotRefGapCutHols0xc34bb280x42e9b20 mesh  nv    144 nf    304
     i  62 aidx  62 midx  62 name                       BotRefHols0xc3cd3800x42ea7c0 mesh  nv    144 nf    304

     i  65 aidx  65 midx  65 name                 SstBotCirRibBase0xc26e2d00x42eba20 mesh  nv     16 nf     28
     i  69 aidx  69 midx  69 name                 SstTopCirRibBase0xc264f780x42ee5b0 mesh  nv     34 nf     64

     i 105 aidx 105 midx 105 name                  led-source-assy0xc3061d00x42fa480 mesh  nv   1022 nf   2016
     i 112 aidx 112 midx 112 name                      source-assy0xc2d5d780x42fda80 mesh  nv   1022 nf   2016
     i 132 aidx 132 midx 132 name              amcco60-source-assy0xc0b1df80x43026c0 mesh  nv   1022 nf   2016
     i 140 aidx 140 midx 140 name                        LsoOflTnk0xc17d9280x4303ab0 mesh  nv    488 nf    976
     i 142 aidx 142 midx 142 name                        GdsOflTnk0xc3d51600x4305940 mesh  nv    880 nf   1760
     i 145 aidx 145 midx 145 name                  OflTnkContainer0xc17cf500x4307970 mesh  nv    344 nf    672

     i 200 aidx 200 midx 200 name                  table_panel_box0xc00f5580x430c870 mesh  nv     58 nf    116
     i 211 aidx 211 midx 211 name    PLACEHOLDER_near_pool_iws_box0xc288ce80x430fa20 mesh  nv     36 nf     12
     i 213 aidx 213 midx 213 name            near_pool_curtain_box0xc2cef480x4310760 mesh  nv     34 nf     64

     i 232 aidx 232 midx 232 name    PLACEHOLDER_near_pool_ows_box0xbf8c8a80x4317310 mesh  nv     36 nf     12
     i 234 aidx 234 midx 234 name              near_pool_liner_box0xc2dcc280x4318080 mesh  nv     34 nf     64
     i 236 aidx 236 midx 236 name               near_pool_dead_box0xbf8a2800x4318dd0 mesh  nv     34 nf     64

     i 245 aidx 245 midx 245 name               near-radslab-box-90xcd31ea00x4319f60 mesh  nv     34 nf     64

    2019-09-19 18:42:23.584 ERROR [107750] [GMeshLib::getMeshSimple@321]  mesh indices do not match  m_meshes index 249 mesh.index 0
     i 249 aidx   0 midx   0 name               near_top_cover_box0xc23f9700x42cebb0 mesh  nv     34 nf     64
    2019-09-19 18:42:23.584 ERROR [107750] [GMeshLib::getMeshSimple@321]  mesh indices do not match  m_meshes index 250 mesh.index 56
     i 250 aidx  56 midx  56 name                 RadialShieldUnit0xc3d7da80x42db120 mesh  nv    304 nf    628
    2019-09-19 18:42:23.584 ERROR [107750] [GMeshLib::getMeshSimple@321]  mesh indices do not match  m_meshes index 251 mesh.index 57
     i 251 aidx  57 midx  57 name                    TopESRCutHols0xbf9de100x42e5980 mesh  nv    578 nf   1188
    2019-09-19 18:42:23.584 ERROR [107750] [GMeshLib::getMeshSimple@321]  mesh indices do not match  m_meshes index 252 mesh.index 58
     i 252 aidx  58 midx  58 name                 TopRefGapCutHols0xbf9cef80x42e6860 mesh  nv    296 nf    608
    ...
    2019-09-19 18:42:23.584 ERROR [107750] [GMeshLib::getMeshSimple@321]  mesh indices do not match  m_meshes index 253 mesh.index 59
    i 269 aidx 234 midx 234 name              near_pool_liner_box0xc2dcc280x4318080 mesh  nv     34 nf     64
    2019-09-19 18:42:23.585 ERROR [107750] [GMeshLib::getMeshSimple@321]  mesh indices do not match  m_meshes index 270 mesh.index 236
     i 270 aidx 236 midx 236 name               near_pool_dead_box0xbf8a2800x4318dd0 mesh  nv     34 nf     64
    2019-09-19 18:42:23.585 ERROR [107750] [GMeshLib::getMeshSimple@321]  mesh indices do not match  m_meshes index 271 mesh.index 245
     i 271 aidx 245 midx 245 name               near-radslab-box-90xcd31ea00x4319f60 mesh  nv     34 nf     64



Update the key, the digest is the same but the world pointer changed::

    2019-09-19 18:42:23.776 INFO  [107750] [Opticks::reportGeoCacheCoordinates@933]  (envvar) OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x5d42890_PV.5aa828335373870398bf4f738781da6c
    2019-09-19 18:42:23.776 INFO  [107750] [Opticks::reportGeoCacheCoordinates@934]  (live)   OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x4552410_PV.5aa828335373870398bf4f738781da6c


Try to visualize with recreated geocache::

    OKTest 
      * cannot find geometry, world too large problem
      * starting the propagation animation shows are positioned at photon source position

    OKTest --geocenter
      * the source is at the center of the world box, which for DYB is a bizarre location off in nowhere  

    OKTest --target 2 --xanalytic
      * find analytic geometry, but the test photon source is strangely placed   
      * no geo selection, see RPC and other above pool volumes 

    OKTest --target 3152 --xanalytic 
      * familiar DYB viewpoint
      * photon source still at center of world box, way off nowhere 




Repeat the export and place the GDML into more permanent location
--------------------------------------------------------------------------


::

    export OPTICKS_LEGACY_GEOMETRY_ENABLED=1 

    ini

    OKTest   # check get the usual DYB geometry via legacy route, resource handling revolves around G4DAE path in legacy 

    OKG4Test --export    # OKG4Test uses CG4 functionality to load the old GDML, merge in info from G4DAE

    tmp
    [blyth@localhost tmp]$ diff CGeometry.Sep19/CGeometry.gdml CGeometry/CGeometry.gdml   ## pointer refs everywhere so diff not informative



Promote that to a more permanent location::

    opticksaux-dx-
    /home/blyth/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export

    p=$(opticksaux-dx-)_v0.gdml
    mkdir -p $(dirname $p)
    cp $TMP/CGeometry/CGeometry.gdml $p

    [blyth@localhost tmp]$ l $p
    -rw-rw-r--. 1 blyth blyth 4544579 Sep 23 13:46 /home/blyth/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v0.gdml


Functions for this version::

    geocache-dx-v0-(){  opticksdata- ; geocache-create- --gdmlpath $(opticksaux-dx-)_v0.gdml --x4polyskip 211,232  --geocenter $* ; }     
    geocache-dx-v0-comment(){ echo export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst ; }     
    geocache-dx-v0(){   geocache-dx-v0- --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }  

    * --geocenter avoids black screen viz, but the world box too large problem is apparent 


Create geocache:

1. vip: switch off legacy in environment:: 

    150 unset OPTICKS_LEGACY_GEOMETRY_ENABLED 
    151 #export OPTICKS_LEGACY_GEOMETRY_ENABLED=1

2. ini ; eo 

3. geocache- ; geocache-dx-v0

4. record the new live key::

    2019-09-23 13:55:49.399 INFO  [114990] [OpticksProfile::accumulateDump@287] Opticks::postgeocache nacc 0
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::reportGeoCacheCoordinates@920]  ok.idpath  /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::reportGeoCacheCoordinates@921]  ok.keyspec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::reportGeoCacheCoordinates@922]  To reuse this geometry: 
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::reportGeoCacheCoordinates@923]    1. set envvar OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::reportGeoCacheCoordinates@924]    2. enable envvar sensitivity with --envkey argument to Opticks executables 
    2019-09-23 13:55:49.399 FATAL [114990] [Opticks::reportGeoCacheCoordinates@932] THE LIVE keyspec DOES NOT MATCH THAT OF THE CURRENT ENVVAR 
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::reportGeoCacheCoordinates@933]  (envvar) OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x4552410_PV.5aa828335373870398bf4f738781da6c
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::reportGeoCacheCoordinates@934]  (live)   OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c
    2019-09-23 13:55:49.399 INFO  [114990] [Opticks::dumpRC@227]  rc 0 rcmsg : -

::

    382 geocache-dx-v0-key(){ echo OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c ; }

    export OPTICKS_KEY=$(geocache-;geocache-dx-v0-key)  

Tidy up OPTICKS_KEY envvar setup in .opticks_setup::

    unset OPTICKS_KEY
    #export OPTICKS_KEY=$(geocache-;geocache-j1808-v5-key)
    export OPTICKS_KEY=$(geocache-;geocache-dx-v0-key) 


kcd::

    [blyth@localhost ~]$ kcd
    /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    rundate
    20190923_135502
    runstamp
    1569218102
    argline
    /home/blyth/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v0.gdml --x4polyskip 211,232 --geocenter --runfolder geocache-dx-v0 --runcomment export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst 
    runcomment
    export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst
    runlabel
    R0_cvd_1
    runfolder
    geocache-dx-v0



Copy the geocache just created from P to C::

   ssh C

   (base) [blyth@gilda03 1]$ p=/home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
   (base) [blyth@gilda03 1]$ l $p
   ls: cannot access /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1: No such file or directory
   (base) [blyth@gilda03 1]$ mkdir -p $(dirname $p) ; scp -r P:$p $(dirname $p)/


opticks-t check
----------------------

Gold::

    SLOW: tests taking longer that 15 seconds
      7  /34  Test #7  : CFG4Test.CG4Test                              Passed                         17.09  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         21.53  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      17.60  

    FAILS:  6   / 416   :  Mon Sep 23 14:36:27 2019   
      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     4.94   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     4.91   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     5.03   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     4.91   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     4.92   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      17.60  
    [blyth@localhost opticks]$ 

Mostly Bialkali sensor surface assert presumably. Skipping for non-legacy in CPropLib makes all but IntegrationTests.tboolean.box pass::

    +    bool legacy = Opticks::IsLegacyGeometryEnabled() ; 
    +    if(is_sensor_material && legacy)
         {
             addSensorMaterialProperties(mpt, name ) ; 
         }
    +

IntegrationTests.tboolean.box failing for lack of parameter.json in Profile.loadMeta.


Silver::

    SLOW: tests taking longer that 15 seconds

    FAILS:  14  / 416   :  Mon Sep 23 14:36:07 2019   
      18 /25  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      7.00   
      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     5.94   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     5.62   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         5.66   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     5.58   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     5.59   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         5.66   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         5.73   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     5.62   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         5.67   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     5.66   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     5.71   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         5.80   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      13.95  







 

TODO
------

* how to target a sensible volume for the photon source ?
* adapt old geo selection functionality from legacy to geocache route  ?

  * :doc:`geoselection-in-new-geometry-workflow`

* for testing on silver and gold using a common /cvmfs/opticks.ihep.ac.cn/ok/shared/geocache/ would be useful

  * before real /cvmfs can fake it with rsync 
  * :doc:`cvmfs-for-shared-geocache-would-be-useful`


Thinking about copying this geocache to shared for "simon" testing
-------------------------------------------------------------------------

::

    [blyth@localhost geocache]$ l
    total 0
    drwxrwxr-x. 3 blyth blyth 23 Sep 19 18:41 OKX4Test_World0xc15cfc00x4552410_PV_g4live
    drwxrwxr-x. 3 blyth blyth 23 Sep 18 22:22 OKX4Test_World0xc15cfc00x5d42890_PV_g4live
    drwxrwxr-x. 3 blyth blyth 23 Aug 30  2018 OKX4Test_lWorld0x4bc2710_PV_g4live
    drwxrwxr-x. 3 blyth blyth 23 Aug 23  2018 CerenkovMinimal_World_g4live
    drwxrwxr-x. 3 blyth blyth 23 Aug 12  2018 OKX4Test_World0xc15cfc0_PV_g4live
    drwxrwxr-x. 3 blyth blyth 23 Jul 16  2018 X4PhysicalVolumeTest_World_g4live
    drwxrwxr-x. 4 blyth blyth 36 Jul  6  2018 DayaBay_VGDX_20140414-1300
    [blyth@localhost geocache]$ pwd
    /home/blyth/.opticks/geocache


Its not appropriate to do that due to poor audit control of this geocache, the gdmlpath is 
not a permanent one : so this geocache is not easily reproducible. Need to mint 
a proper "permanent" GDML path and then recreate the geocache with a bash function
named for this purpose.

::

    blyth@localhost opticks]$ ini
    [blyth@localhost opticks]$ kcd
    /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x4552410_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    rundate
    20190919_184141
    runstamp
    1568889701
    argline
    /home/blyth/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/tmp/CGeometry/CGeometry.gdml --runfolder geocache-dxtmp --runcomment gdml-insitu-created-by-OKG4Test-export --x4polyskip 211,232 
    runcomment
    gdml-insitu-created-by-OKG4Test-export
    runlabel
    R0_cvd_1
    runfolder
    geocache-dxtmp
    [blyth@localhost 1]$ 











check legacy opticks-t with default old DYB geometry
-------------------------------------------------------------

Switch back to legacy::

    FAILS:  3   / 415   :  Thu Sep 19 19:46:07 2019   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.27   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     1.39   
          known  

      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      15.27  
          analysis fail from commenting  WITH_LOGDOUBLE 



non-legacy mode opticks-t with the new tmp geocache and key
-------------------------------------------------------------

::

    FAILS:  17  / 415   :  Thu Sep 19 19:41:54 2019   
      3  /3   Test #3  : AssimpRapTest.AssimpGGeoTest                  Child aborted***Exception:     0.10   
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                Child aborted***Exception:     0.09   

         skip these fail to access DAE with message :
               this test is not relevant to non-legacy running and will be skipped in future 

      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.25   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     1.30   

          known  

      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     5.12   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     4.97   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     5.83   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     5.68   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     5.78   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     5.81   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     5.79   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     5.02   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     5.87   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     5.04   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     5.00   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     5.82   

          all these 12 cause by the below two issues   
          

      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      15.05  
           same analysis fail   


    

non-legacy OKG4Test name match fail from DYB prefix
----------------------------------------------------------

* fixed by comparing unprefixed names for g4 materials with names 
  prefixed with "/dd/Materials/"

::

    CTestDetectorTest
    CGDMLDetectorTest
    CGeometryTest    
    CG4Test
    CInterpolationTest
    CRandomEngineTest  
    OKG4Test
 


::

    2019-09-19 20:18:34.417 FATAL [265718] [CMaterialSort::sort@83]  sorting G4MaterialTable using order kv 36
    2019-09-19 20:18:34.624 ERROR [265718] [CGDMLDetector::addMPTLegacyGDML@175]  Looks like GDML has succeded to load material MPTs   nmat 36 nmat_without_mpt 0 skipping the fixup 
    2019-09-19 20:18:34.624 INFO  [265718] [CGDMLDetector::standardizeGeant4MaterialProperties@240] [
    CGeometryTest: /home/blyth/opticks/extg4/X4MaterialLib.cc:126: void X4MaterialLib::init(): Assertion `name_match' failed.
    Aborted (core dumped)


::

    (gdb) bt
    #0  0x00007fffe1fc5207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe1fc68f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe1fbe026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe1fbe0d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff492cac8 in X4MaterialLib::init (this=0x7fffffffc480) at /home/blyth/opticks/extg4/X4MaterialLib.cc:126
    #5  0x00007ffff492c635 in X4MaterialLib::X4MaterialLib (this=0x7fffffffc480, mtab=0x7fffed79c0c0 <G4Material::theMaterialTable>, mlib=0x6cc640) at /home/blyth/opticks/extg4/X4MaterialLib.cc:81
    #6  0x00007ffff492c5fb in X4MaterialLib::Standardize (mtab=0x7fffed79c0c0 <G4Material::theMaterialTable>, mlib=0x6cc640) at /home/blyth/opticks/extg4/X4MaterialLib.cc:72
    #7  0x00007ffff492c5d1 in X4MaterialLib::Standardize () at /home/blyth/opticks/extg4/X4MaterialLib.cc:67
    #8  0x00007ffff4c7ecc1 in CGDMLDetector::standardizeGeant4MaterialProperties (this=0x5ca99d0) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:241
    #9  0x00007ffff4c7e249 in CGDMLDetector::init (this=0x5ca99d0) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:106
    #10 0x00007ffff4c7ddef in CGDMLDetector::CGDMLDetector (this=0x5ca99d0, hub=0x6b9880, query=0x6c58b0, sd=0x5ca7370) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:63
    #11 0x00007ffff4c24b12 in CGeometry::init (this=0x5ca9920) at /home/blyth/opticks/cfg4/CGeometry.cc:99
    #12 0x00007ffff4c24924 in CGeometry::CGeometry (this=0x5ca9920, hub=0x6b9880, sd=0x5ca7370) at /home/blyth/opticks/cfg4/CGeometry.cc:82
    #13 0x00007ffff4c9710c in CG4::CG4 (this=0x5ac5c50, hub=0x6b9880) at /home/blyth/opticks/cfg4/CG4.cc:155
    #14 0x00007ffff7bd453f in OKG4Mgr::OKG4Mgr (this=0x7fffffffd5b0, argc=1, argv=0x7fffffffd8f8) at /home/blyth/opticks/okg4/OKG4Mgr.cc:107
    #15 0x000000000040399a in main (argc=1, argv=0x7fffffffd8f8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:27
    (gdb) f 4
    #4  0x00007ffff492cac8 in X4MaterialLib::init (this=0x7fffffffc480) at /home/blyth/opticks/extg4/X4MaterialLib.cc:126
    126         assert(name_match ); 
    (gdb) list
    121                 << " index " << i 
    122                 << " pmap_name " << pmap_name
    123                 << " m4_name " << m4_name
    124                 ;
    125 
    126         assert(name_match ); 
    127         if( m4->GetMaterialPropertiesTable() == NULL ) continue ; 
    128 
    129         G4MaterialPropertiesTable* mpt = X4PropertyMap::Convert( pmap ) ; 
    130         m4->SetMaterialPropertiesTable( mpt ) ; 
    (gdb) p pmap_name
    $1 = 0x6d3078 "PPE"
    (gdb) p m4_name
    $2 = "/dd/Materials/PPE"
    (gdb) 


::

    2019-09-19 20:14:51.907 INFO  [260276] [X4MaterialLib::init@111]     0 okmat                            PPE g4mat              /dd/Materials/PPE
    2019-09-19 20:14:51.907 FATAL [260276] [X4MaterialLib::init@119]  MATERIAL NAME MISMATCH  index 0 pmap_name PPE m4_name /dd/Materials/PPE
    CRandomEngineTest: /home/blyth/opticks/extg4/X4MaterialLib.cc:126: void X4MaterialLib::init(): Assertion `name_match' failed.
    Aborted (core dumped)



CFG4 tests : Bialkali surf assert
-------------------------------------------

After fixing the name prefix mismatch, cfg4-t gives five fails all from the same cause::

    85% tests passed, 5 tests failed out of 34

    Total Test time (real) =  78.83 sec

    The following tests FAILED:
          1 - CFG4Test.CMaterialLibTest (Child aborted)
          2 - CFG4Test.CMaterialTest (Child aborted)
         25 - CFG4Test.CGROUPVELTest (Child aborted)
         32 - CFG4Test.CCerenkovGeneratorTest (Child aborted)
         33 - CFG4Test.CGenstepSourceTest (Child aborted)
    Errors while running CTest
    Thu Sep 19 20:48:13 CST 2019
    [blyth@localhost cfg4]$ 


::

    2019-09-19 20:10:16.368 INFO  [253270] [CMaterialLib::convert@172]  g4mat 0x5b10c50 name             LiquidScintillator Pmin  1.512e-06 Pmax 2.0664e-05 Wmin         60 Wmax        820
    2019-09-19 20:10:16.368 FATAL [253270] [CPropLib::makeMaterialPropertiesTable@263] CPropLib::makeMaterialPropertiesTable material with SENSOR_MATERIAL name Bialkali but no sensor_surface 
    2019-09-19 20:10:16.368 FATAL [253270] [CPropLib::makeMaterialPropertiesTable@267] m_sensor_surface is obtained from slib at CPropLib::init  when Bialkai material is in the mlib  it is required for a sensor surface (with EFFICIENCY/detect) property  to be in the slib 
    CGenstepSourceTest: /home/blyth/opticks/cfg4/CPropLib.cc:273: G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial*): Assertion `surf' failed.
    
    Program received signal SIGABRT, Aborted.
    0x00007fffe4e78207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe4e78207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe4e798f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe4e71026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe4e710d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7ae03ce in CPropLib::makeMaterialPropertiesTable (this=0x5abbbd0, ggmat=0x6f3cf0) at /home/blyth/opticks/cfg4/CPropLib.cc:273
    #5  0x00007ffff7af24a3 in CMaterialLib::convertMaterial (this=0x5abbbd0, kmat=0x6f3cf0) at /home/blyth/opticks/cfg4/CMaterialLib.cc:260
    #6  0x00007ffff7af1819 in CMaterialLib::convert (this=0x5abbbd0) at /home/blyth/opticks/cfg4/CMaterialLib.cc:152
    #7  0x000000000040453b in main (argc=1, argv=0x7fffffffd8e8) at /home/blyth/opticks/cfg4/tests/CGenstepSourceTest.cc:60
    (gdb) 



::

    2019-09-19 20:12:23.329 FATAL [256392] [CPropLib::makeMaterialPropertiesTable@263] CPropLib::makeMaterialPropertiesTable material with SENSOR_MATERIAL name Bialkali but no sensor_surface 
    2019-09-19 20:12:23.329 FATAL [256392] [CPropLib::makeMaterialPropertiesTable@267] m_sensor_surface is obtained from slib at CPropLib::init  when Bialkai material is in the mlib  it is required for a sensor surface (with EFFICIENCY/detect) property  to be in the slib 
    CCerenkovGeneratorTest: /home/blyth/opticks/cfg4/CPropLib.cc:273: G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial*): Assertion `surf' failed.
    
    (gdb) bt
    #4  0x00007ffff7ae03ce in CPropLib::makeMaterialPropertiesTable (this=0x5abac00, ggmat=0x6f2d20) at /home/blyth/opticks/cfg4/CPropLib.cc:273
    #5  0x00007ffff7af24a3 in CMaterialLib::convertMaterial (this=0x5abac00, kmat=0x6f2d20) at /home/blyth/opticks/cfg4/CMaterialLib.cc:260
    #6  0x00007ffff7af1819 in CMaterialLib::convert (this=0x5abac00) at /home/blyth/opticks/cfg4/CMaterialLib.cc:152
    #7  0x000000000040433b in main (argc=1, argv=0x7fffffffd8d8) at /home/blyth/opticks/cfg4/tests/CCerenkovGeneratorTest.cc:71
    (gdb) 




Search for Bialkali in issues
--------------------------------

* :doc:`direct_route_needs_AssimpGGeo_convertSensors_equivalent`

Hmm this looks to be an involved issue, with lots of history to review.



