hans_Opticks_patch_for_1100_against_github_017_tag
====================================================


Issue
--------

Hi Simon


Geant4 11.0 has been released last Friday and it includes CaTS as an advanced example with using opticks
as a build/runtime option. To make opticks compile with the new Geant4 version I had to  modify the files attached.
Alternatively I forked
https://github.com/simoncblyth/opticks and created a new branch/tag v0.1.7-brnch/v0.1.7 which contains the changes.

Still I have some test failing and I am looking into it.::

    ./local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath CaTS/gdml/simpleLArTPC.gdml
    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World_PV.ae2396fd17e7de276925c54596aad9fc
    opticks-t

gives::

    AILS:  14  / 496   :  Thu Dec 16 10:51:10 2021  
      5  /35  Test #5  : OptiXRapTest.Roots3And4Test                   Subprocess aborted***Exception:   0.18  
      25 /35  Test #25 : OptiXRapTest.interpolationTest                ***Failed                      0.49  
      32 /35  Test #32 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Subprocess aborted***Exception:   0.24  
      1  /44  Test #1  : CFG4Test.CMaterialLibTest                     Subprocess aborted***Exception:   0.16  
      2  /44  Test #2  : CFG4Test.CMaterialTest                        Subprocess aborted***Exception:   0.17  
      3  /44  Test #3  : CFG4Test.CTestDetectorTest                    Subprocess aborted***Exception:   0.18  
      5  /44  Test #5  : CFG4Test.CGDMLDetectorTest                    Subprocess aborted***Exception:   0.16  
      7  /44  Test #7  : CFG4Test.CGeometryTest                        Subprocess aborted***Exception:   0.16  
      14 /44  Test #14 : CFG4Test.G4MaterialPropertiesTableTest        Subprocess aborted***Exception:   0.12  
      15 /44  Test #15 : CFG4Test.CMPTTest                             Subprocess aborted***Exception:   0.13  
      26 /44  Test #26 : CFG4Test.CInterpolationTest                   Subprocess aborted***Exception:   0.18  
      28 /44  Test #28 : CFG4Test.CGROUPVELTest                        Subprocess aborted***Exception:   0.19  
      36 /44  Test #36 : CFG4Test.CCerenkovGeneratorTest               Subprocess aborted***Exception:   0.16  
      37 /44  Test #37 : CFG4Test.CGenstepSourceTest                   Subprocess aborted***Exception:   0.16  


also trying the daya bay gdml file to create the geocache seems to end in an endless loop.::

    ./local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache \
         --gdmlpath /data2/wenzel/gputest_11.0/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml

Cheers Hans 



Explode attached tarball
-------------------------

:: 

    epsilon:~ blyth$ cd /tmp
    epsilon:tmp blyth$ tar ztvf opticks_G4_11_0.tgz
    -rw-rw-r--  0 wenzel wenzel   4990 Dec 10 19:14 cfg4/CDump.cc
    -rw-rw-r--  0 wenzel wenzel  20004 Dec 10 19:11 cfg4/CMPT.cc
    -rw-rw-r--  0 wenzel wenzel  19610 Dec 13 21:22 cfg4/CPropLib.cc
    -rw-rw-r--  0 wenzel wenzel   7985 Dec 14 03:12 cfg4/CTestDetector.cc
    -rw-rw-r--  0 wenzel wenzel   5383 Dec 13 16:11 cfg4/tests/CGDMLPropertyTest.cc
    -rw-rw-r--  0 wenzel wenzel   7351 Dec 10 19:57 cfg4/tests/CInterpolationTest.cc
    -rw-rw-r--  0 wenzel wenzel   1819 Dec 10 19:28 cfg4/tests/CMakeLists.txt
    -rw-rw-r--  0 wenzel wenzel   3322 Dec 13 16:10 cfg4/tests/G4MaterialPropertiesTableTest.cc
    -rw-rw-r--  0 wenzel wenzel   3022 Dec 10 18:08 cmake/Modules/OpticksCXXFlags.cmake
    -rw-rw-r--  0 wenzel wenzel   4971 Dec 13 16:12 extg4/X4Dump.cc
    -rw-rw-r--  0 wenzel wenzel   5229 Dec 14 02:55 extg4/X4MaterialLib.cc
    -rw-rw-r--  0 wenzel wenzel   6944 Dec 13 16:07 extg4/X4MaterialPropertiesTable.cc
    -rw-rw-r--  0 wenzel wenzel  11491 Dec 13 16:09 extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
    -rw-rw-r--  0 wenzel wenzel   2944 Dec 10 18:07 optickscore/OpticksSwitches.h
    -rw-r--r--  0 wenzel wenzel  23418 Dec 14 16:53 CaTS/gdml/simpleLArTPC.gdml
    epsilon:tmp blyth$ 


Huh, I only go to v0.1.6::

    epsilon:opticks blyth$ git tag
    v0.0.0-rc1
    v0.0.0-rc2
    v0.0.0-rc3
    v0.1.0-rc1
    v0.1.0-rc2
    v0.1.1
    v0.1.2
    v0.1.3
    v0.1.4
    v0.1.5
    v0.1.6


* https://github.com/simoncblyth/opticks/tags


Compare with latest Opticks
-----------------------------



::

    epsilon:tmp blyth$ mkdir /tmp/hans 
    epsilon:tmp blyth$ mv /tmp/opticks_G4_11_0.tgz /tmp/hans
    epsilon:tmp blyth$ cd /tmp/hans
    epsilon:hans blyth$ 
    epsilon:hans blyth$ 
    epsilon:hans blyth$ tar zxvf opticks_G4_11_0.tgz
    x cfg4/CDump.cc
    x cfg4/CMPT.cc
    x cfg4/CPropLib.cc
    x cfg4/CTestDetector.cc
    x cfg4/tests/CGDMLPropertyTest.cc
    x cfg4/tests/CInterpolationTest.cc
    x cfg4/tests/CMakeLists.txt
    x cfg4/tests/G4MaterialPropertiesTableTest.cc
    x cmake/Modules/OpticksCXXFlags.cmake
    x extg4/X4Dump.cc
    x extg4/X4MaterialLib.cc
    x extg4/X4MaterialPropertiesTable.cc
    x extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
    x optickscore/OpticksSwitches.h
    x CaTS/gdml/simpleLArTPC.gdml
    epsilon:hans blyth$ 




::

    epsilon:hans blyth$ find . -type f -exec echo diff  {} ~/opticks/{} \; 
    diff ./CaTS/gdml/simpleLArTPC.gdml /Users/blyth/opticks/./CaTS/gdml/simpleLArTPC.gdml

    diff ./cfg4/CPropLib.cc /Users/blyth/opticks/./cfg4/CPropLib.cc
    diff ./cfg4/tests/CGDMLPropertyTest.cc /Users/blyth/opticks/./cfg4/tests/CGDMLPropertyTest.cc
    diff ./cfg4/tests/CMakeLists.txt /Users/blyth/opticks/./cfg4/tests/CMakeLists.txt

    diff ./cfg4/tests/G4MaterialPropertiesTableTest.cc /Users/blyth/opticks/./cfg4/tests/G4MaterialPropertiesTableTest.cc
    diff ./cfg4/tests/CInterpolationTest.cc /Users/blyth/opticks/./cfg4/tests/CInterpolationTest.cc
    diff ./cfg4/CDump.cc /Users/blyth/opticks/./cfg4/CDump.cc
    diff ./cfg4/CMPT.cc /Users/blyth/opticks/./cfg4/CMPT.cc
    diff ./cfg4/CTestDetector.cc /Users/blyth/opticks/./cfg4/CTestDetector.cc
    diff ./cmake/Modules/OpticksCXXFlags.cmake /Users/blyth/opticks/./cmake/Modules/OpticksCXXFlags.cmake
    diff ./extg4/X4Dump.cc /Users/blyth/opticks/./extg4/X4Dump.cc
    diff ./extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc /Users/blyth/opticks/./extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
    diff ./extg4/X4MaterialLib.cc /Users/blyth/opticks/./extg4/X4MaterialLib.cc
    diff ./extg4/X4MaterialPropertiesTable.cc /Users/blyth/opticks/./extg4/X4MaterialPropertiesTable.cc
    diff ./opticks_G4_11_0.tgz /Users/blyth/opticks/./opticks_G4_11_0.tgz
    diff ./optickscore/OpticksSwitches.h /Users/blyth/opticks/./optickscore/OpticksSwitches.h
    epsilon:hans blyth$ 



::

    epsilon:hans blyth$ diff ./cfg4/tests/CMakeLists.txt /Users/blyth/opticks/./cfg4/tests/CMakeLists.txt
    18c18
    < #    G4StringTest.cc
    ---
    >     G4StringTest.cc
    epsilon:hans blyth$ 




cfg4/CDump.cc no significant change::

    epsilon:cfg4 blyth$ diff CDump.cc ~/opticks/cfg4/CDump.cc
    60d59
    <         bool warning = false ; 

cfg4/CMPT.cc parallel changes::

    epsilon:cfg4 blyth$ diff CMPT.cc ~/opticks/cfg4/CMPT.cc
    33a34
    > #include "X4MaterialPropertiesTable.hh"
    100d100
    <     G4bool warning ; 
    167d166
    <     G4bool warning ; 
    175c174
    <         MPV* pvec = const_cast<G4MaterialPropertiesTable*>(m_mpt)->GetProperty(pidx );  
    ---
    >         MPV* pvec = const_cast<G4MaterialPropertiesTable*>(m_mpt)->GetProperty(pidx);  
    202,204c201,202
    <     G4String skey(lkey); 
    <     G4int keyIdx = mpt->GetPropertyIndex(skey); 
    <     G4bool createNewKey = keyIdx == -1  ; 
    ---
    >     bool exists = X4MaterialPropertiesTable::PropertyExists(mpt, lkey); 
    >     G4bool createNewKey = exists == false ; 
    220,221c218,229
    <     G4int keyIdx = mpt->GetConstPropertyIndex(skey); 
    <     G4bool createNewKey = keyIdx == -1 ; 
    ---
    > 
    >     // 1st try: nope throws exception for non-existing key 
    >     //G4int keyIdx = mpt->GetConstPropertyIndex(skey);  
    >     //G4bool createNewKey = keyIdx == -1 ; 
    > 
    >     // 2nd try: nope again throws exception from GetConstPropertyIndex just like above
    >     //G4bool exists = mpt->ConstPropertyExists(lkey); 
    >     //G4bool createNewKey = !exists ; 
    >    
    >     // 3rd try 
    >     bool exists = X4MaterialPropertiesTable::ConstPropertyExists(mpt, lkey); 
    >     G4bool createNewKey = !exists ; 
    248d255
    <     G4bool warning ;
    284d290
    <     G4bool warning ; 
    341d346
    <     G4bool warning ; 
    580a586,587
    > Subsequently Geant4 changes G4MaterialPropertiesTable to throw exceptions for non-existing keys
    > 
    585,586c592
    <     G4bool warning = false ; 
    <     G4int index = m_mpt->GetPropertyIndex(key) ;   // this avoids ticking 91072 bug when the key is non-existing
    ---
    >     int index = X4MaterialPropertiesTable::GetPropertyIndex(m_mpt, key) ;   // this avoids ticking 91072 bug when the key is non-existing
    epsilon:cfg4 blyth$ vimdiff CMPT.cc ~/opticks/cfg4/CMPT.cc
    2 files to edit
    epsilon:cfg4 blyth$ 




cfg4/CPropLib.cc 

::

    epsilon:cfg4 blyth$ diff CPropLib.cc ~/opticks/cfg4/CPropLib.cc 
    14,15c14,15
    <  * distributed under the License is distributed on an "AS IS" BASIS,  
    < * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
    ---
    >  * distributed under the License is distributed on an "AS IS" BASIS, 
    >  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
    51a52,53
    > #include "X4MaterialPropertiesTable.hh"
    > 
    266c268
    <     addProperties(mpt, _ggmat, "RINDEX,ABSLENGTH,RAYLEIGH,GROUPVEL");
    ---
    >     addProperties(mpt, _ggmat, "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB,GROUPVEL");
    370,371c372
    <     bool constant = false ;
    < #if G4VERSION_NUMBER < 1100
    ---
    >     bool constant = false ; 
    374,378d374
    < #else
    <     addProperties(mpt, scintillator, "SCINTILLATIONCOMPONENT1,SCINTILLATIONCOMPONENT2", keylocal, constant);
    <     addProperties(mpt, scintillator, "SCINTILLATIONYIELD1,SCINTILLATIONYIELD2,SCINTILLATIONTIMECONSTANT1,SCINTILLATIONTIMECONSTANT2", keylocal, constant ); // this used constant=true formerly
    < #endif
    < 
    540,541c536
    <     G4String skey(lkey); 
    <     G4int keyIdx = mpt->GetPropertyIndex(skey); 
    ---
    >     int keyIdx = X4MaterialPropertiesTable::GetPropertyIndex(mpt, lkey); 
    558a554,562
    > 
 

* looks like CPropLib and CMaterialLib is no longer in critical use (not used from extg4 x4)
  so should try to eliminate this code

::
   
    epsilon:opticks blyth$ opticks-f CPropLib | grep -v CPropLib.cc | grep -v CPropLib.hh
    ...
    ./cfg4/CTestDetector.hh:class CPropLib ; 
    ./cfg4/CMaterialLib.cc:    CPropLib(hub, 0),
    ./cfg4/CMaterialLib.hh:G4The GGeo gets loaded on initializing base class CPropLib.
    ./cfg4/CMaterialLib.hh:class CFG4_API CMaterialLib : public CPropLib 

* perhaps the change in properties is a knee jerk response for new geant4, not due to any particular need 
* if this code does turn out to still be necessary will need to make the keys configurable as which 
   scintillator properties are important to people is variable 




CTestDetector.cc no significant change::

    epsilon:cfg4 blyth$ diff CTestDetector.cc ~/opticks/cfg4/CTestDetector.cc
    130c130
    < q    {
    ---
    >     {


::

    epsilon:cfg4 blyth$ diff tests/CGDMLPropertyTest.cc ~/opticks/cfg4/tests/CGDMLPropertyTest.cc
    149c149
    <     const char* path = SPath::Resolve(path_); 
    ---
    >     const char* path = SPath::Resolve(path_, 0); 
    epsilon:cfg4 blyth$ 


