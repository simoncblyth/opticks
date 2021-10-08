Geant4_1100_needs_cpp17
============================


Hi Soon, Hans, 

> I successfully built and tested opticks v0.1.6 with Gent4.10.7.ref08 on our cluster,
> of which opticks-t results look reasonable (see the test summary included at the bottom)
> - ignoring the last failure which is due to the bug in GetMaterialConstPropertyNames()
> that you found.  Thanks again for updating all opticks codes!

For me opticks-t with the first tarball you provided gave only one FAIL (G4MaterialPropertiesTableTest)
Latest opticks with the first tarball is giving the same::


    SLOW: tests taking longer that 15 seconds
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         35.85  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         35.43  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Passed                         35.45  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Passed                         37.03  

    FAILS:  1   / 499   :  Fri Oct  8 03:41:51 2021   
      15 /45  Test #15 : CFG4Test.G4MaterialPropertiesTableTest        Child aborted***Exception:     0.25   
    (base) [simon@localhost opticks]$ 

The FAIL is from GetProperty not giving nullptr for a non existing key due to the  fMP[-1] bug::

    (gdb) f 4
    #4  0x0000000000403efb in test_GetProperty_NonExisting (mpt_=0x6a1cd0) at /home/simon/opticks/cfg4/tests/G4MaterialPropertiesTableTest.cc:92
    92	    assert( mpv == nullptr ); 
    (gdb) list
    87	    G4MaterialPropertiesTable* mpt = const_cast<G4MaterialPropertiesTable*>(mpt_);   // tut tut GetProperty is not const correct 
    88	    const char* key = "NonExistingKey" ; 
    89	    G4bool warning = false ; 
    90	    G4MaterialPropertyVector* mpv = mpt->GetProperty(key, warning); 
    91	    LOG(info) << " key " << key << " mpv " << mpv ; 
    92	    assert( mpv == nullptr ); 
    93	}
    94	

I recommend that you or Hans investigate the causes of the fails 
by collecting gdb backtraces with context.
You can run those tests individually just using the names of the tests::

    SRngSpecTest
    BDirTest   
    GColorsTest 
    interpolationTest
    X4GDMLBalanceTest 
    G4MaterialPropertiesTableTest 


> Testing opticks v0.1.6 with the master version of Geant4 (today) triggers an
> issue related to the c++17 standard which is the requirement of Geant4 now.
> (see a compilation error, for an example, from G4String.hh in which std::string_view is used).

I see that std::string_view is introduced from c++17 only, 
that being used in G4String.hh forces c++17.

> What is the best way to switch (or migrate) the c++ standard to -std=c++17 
> globally for the opticks installation?  Thanks for your advice.

Currently the standard is fixed in cmake/Modules/OpticksCXXFlags.cmake::

     29      set(CMAKE_CXX_STANDARD 14)
     30      set(CMAKE_CXX_STANDARD_REQUIRED on)
     31 
     32   endif ()

My commit just now bumps that to 17, it om-cleaninstall builds without incident
and the same single test fails. This means that gcc < 5 does not work anymore.  


>    P.S.  You cant get the tarball of the master version of Geant4 (today) by
>    wget https://g4cpt.fnal.gov/g4p/download/geant4.master100721.tar
>    if you want to reproduce the problem.  Again, I modified two files
>    G4Version.hh
>    G4MaterialPropertiesTable.hh   (fix the bug in GetMaterialConstPropertyNames)
>    temporarily.
>
>    1) The result of opticks-t on wc.fnal.gov
>
>    SLOW: tests taking longer that 15 seconds
>
>    FAILS:  6   / 497   :  Thu Oct  7 10:50:58 2021  
>      53 /66  Test #53 : SysRapTest.SRngSpecTest                       Child aborted***Exception:     0.01  
>      2  /39  Test #2  : BoostRapTest.BDirTest                         Child aborted***Exception:     0.01  
>      38 /58  Test #38 : GGeoTest.GColorsTest                          Child aborted***Exception:     0.01  
>      25 /35  Test #25 : OptiXRapTest.interpolationTest                ***Failed                      3.20  
>      21 /35  Test #21 : ExtG4Test.X4GDMLBalanceTest                   Child aborted***Exception:     0.05  
>      15 /45  Test #15 : CFG4Test.G4MaterialPropertiesTableTest        Child aborted***Exception:     0.04  
>
>    2) an example of compilation errors with the master version of Geant4
>
>    Scanning dependencies of target ExtG4
>    [  1%] Building CXX object CMakeFiles/ExtG4.dir/X4CSG.cc.o
>    In file included from /work1/g4gpu/syjun/products/install/geant4.master100721/include/Geant4/globals.hh:50,
>                     from /work1/g4gpu/syjun/products/install/geant4.master100721/include/Geant4/G4ThreeVector.hh:33,
>                     from /work1/g4gpu/syjun/products/src/opticks/extg4/X4Solid.hh:26,
>                     from /work1/g4gpu/syjun/products/src/opticks/extg4/X4CSG.cc:37:
>    /work1/g4gpu/syjun/products/install/geant4.master100721/include/Geant4/G4String.hh:91:31: error: ‘std::string_view’ has not been declared
>       inline G4int compareTo(std::string_view, caseCompare mode = exact) const;
>                                   ^~~~~~~~~~~









* https://github.com/stevenlovegrove/Pangolin/issues/469
