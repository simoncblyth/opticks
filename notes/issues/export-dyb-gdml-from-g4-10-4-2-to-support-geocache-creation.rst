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



