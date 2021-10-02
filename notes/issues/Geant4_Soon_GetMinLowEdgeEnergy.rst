Geant4_Soon_GetMinLowEdgeEnergy
==================================


> Hi Simon,
>
> I was also going to complain to Vladimir for dropping all these public 
> interfaces without a warning (i.e., a warning message through codes 
> before a major release) - on the other hand, I also understand that
> there may be too many warnings in this case and his efforts to clean 
> up as many obsolete methods (to him) as possible.
>
> Regarding to 
> 
> GetMinLowEdgeEnergy() and 
> GetMaxLowEdgeEnergy(),
> 
> I think that they can be replaced by either
> 
> GetMinEnergy() and
> GetMaxEnergy()
> 
> or alternatively
> 
> GetLowEdgeEnergy(binIndex1) and 
> GetLowEdgeEnergy(binIndex2),
> 
> where binIndex1 and binIndex2 is the index of the bin depending on
> which edge energy to get (in this case, may be 0 and numberOfNodes
> (i.e, GetVectorLength()), but please correct me if I guessed wrong).

> which have/ been available in old versions.
>
> Same for IsFilledVectorExist() which can be simply replaced by
> (GetVectorLength()>0)
>
> The wrapper will also work, but you have to change all places anyway passing
> the G4PhysicsFreeVector to the wrapper.  So, it is probably cleaner to replace them
> with existing interfaces which do not depend on the Geant4 version.


Thank you for the suggestions. I agree, moving to unchanging API is the simplest way.  
As for "G4PhysicsVector::GetLowEdgeEnergy(size_t binNumber) const" , that is marked as obsolete 
in 1042 so I plumped for "->Energy(0)" with the below one liners to do all the edits::


   perl -pi -e 's/GetMinLowEdgeEnergy\(\)/Energy(0)/' $(find . -name '*.cc' -exec grep -l GetMinLowEdgeEnergy {} \;)

   perl -pi -e 's/\s(\w*)->GetMaxLowEdgeEnergy\(\)/ $1->Energy($1->GetVectorLength()-1)/' $(find . -name '*.cc' -exec grep -l GetMaxLowEdgeEnergy {} \;)

   perl -pi -e 's/IsFilledVectorExist\(\)/GetVectorLength()>0/' $(find . -name '*.cc' -exec grep -l IsFilledVectorExist {} \;)
    


> For your local test, I put a tar ball of geant4-10-07-ref-08 on my cluster 
> which you can get by
> wget https://g4cpt.fnal.gov/g4p/download/geant4.10.7.r08.tar
> tar -xzf geant4.10.7.r08.tar
>
> in the case that you do not have a direct access to the geant4-dev repository,
> (The G4Version.hh was already modified with 
> #define G4VERSION_NUMBER 91072
> in the tarball so that we can tweak this reference release for mimicking the target 
> version number,1100).
>


I do not have access, not being a Geant4 collaboration member. 
So thank you for the tarball that was useful to check my changes. 


> Please let us know when the next opticks version is available with updates, 
> so that we can test it right away.  Also, hope that this is the last
> hiccup from the Geant4 side. Thanks always!
>
> Regards,
> ---Soon








::

    SLOW: tests taking longer that 15 seconds
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     37.12  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     36.97  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     37.01  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     37.90  


    FAILS:  12  / 497   :  Sun Oct  3 02:18:39 2021   
      9  /35  Test #9  : ExtG4Test.X4MaterialTest                      Child aborted***Exception:     0.16   
      12 /35  Test #12 : ExtG4Test.X4MaterialTableTest                 Child aborted***Exception:     0.18   
      13 /35  Test #13 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.18   
      14 /35  Test #14 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.18   
      34 /35  Test #34 : ExtG4Test.X4SurfaceTest                       Child aborted***Exception:     0.47   
      1  /45  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     3.76   
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     37.01  
      29 /45  Test #29 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     3.17   

      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     37.12  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     36.97  
      15 /45  Test #15 : CFG4Test.G4MaterialPropertiesTableTest        Child aborted***Exception:     0.25   
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     37.90  
    (base) [simon@localhost opticks]$ 




x4 fails
---------

::

     9/35 Test  #9: ExtG4Test.X4MaterialTest ................................Child aborted***Exception:   0.16 sec

    -------- WWWW ------- G4Exception-START -------- WWWW -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat200
          issued by : G4MaterialPropertiesTable::GetConstPropertyIndex()
    Constant Material Property Index for key RINDEX not found.
    *** This is just a warning message. ***
    -------- WWWW ------- G4Exception-END -------- WWWW -------

    X4MaterialTest: /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:155: static void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>*, const G4MaterialPropertiesTable*, char): Assertion `pidx > -1' failed.

          Start 10: ExtG4Test.X4MaterialWaterStandaloneTest
    10/35 Test #10: ExtG4Test.X4MaterialWaterStandaloneTest .................   Passed    0.10 sec
          Start 11: ExtG4Test.X4MaterialWaterTest
    11/35 Test #11: ExtG4Test.X4MaterialWaterTest ...........................   Passed    0.17 sec
          Start 12: ExtG4Test.X4MaterialTableTest
    12/35 Test #12: ExtG4Test.X4MaterialTableTest ...........................Child aborted***Exception:   0.18 sec
    2021-10-03 02:15:35.852 FATAL [259008] [Opticks::envkey@348]  --allownokey option prevents key checking : this is for debugging of geocache creation 
    2021-10-03 02:15:35.857 FATAL [259008] [OpticksResource::init@122]  CAUTION : are allowing no key 

    -------- WWWW ------- G4Exception-START -------- WWWW -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat200
          issued by : G4MaterialPropertiesTable::GetConstPropertyIndex()
    Constant Material Property Index for key RINDEX not found.
    *** This is just a warning message. ***
    -------- WWWW ------- G4Exception-END -------- WWWW -------

    X4MaterialTableTest: /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:155: static void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>*, const G4MaterialPropertiesTable*, char): Assertion `pidx > -1' failed.

          Start 13: ExtG4Test.X4PhysicalVolumeTest
    13/35 Test #13: ExtG4Test.X4PhysicalVolumeTest ..........................Child aborted***Exception:   0.18 sec

    -------- WWWW ------- G4Exception-START -------- WWWW -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat200
          issued by : G4MaterialPropertiesTable::GetConstPropertyIndex()
    Constant Material Property Index for key RINDEX not found.
    *** This is just a warning message. ***
    -------- WWWW ------- G4Exception-END -------- WWWW -------





X4MaterialTest
-----------------

::

    -------- WWWW ------- G4Exception-START -------- WWWW -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat200
          issued by : G4MaterialPropertiesTable::GetConstPropertyIndex()
    Constant Material Property Index for key RINDEX not found.
    *** This is just a warning message. ***
    -------- WWWW ------- G4Exception-END -------- WWWW -------


    (gdb) bt
    #0  0x00007fffeb015387 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb016a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb00e1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeb00e252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b693e4 in X4MaterialPropertiesTable::AddProperties (pmap=0x6d9b40, mpt=0x6d41e0, mode=71 'G') at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:155
    #5  0x00007ffff7b68c0f in X4MaterialPropertiesTable::init (this=0x7fffffffc4e0) at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:54
    #6  0x00007ffff7b68bdb in X4MaterialPropertiesTable::X4MaterialPropertiesTable (this=0x7fffffffc4e0, pmap=0x6d9b40, mpt=0x6d41e0, mode=71 'G')
        at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:49
    #7  0x00007ffff7b68b6a in X4MaterialPropertiesTable::Convert (pmap=0x6d9b40, mpt=0x6d41e0, mode=71 'G') at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:40
    #8  0x00007ffff7b6421a in X4Material::init (this=0x7fffffffc760) at /home/simon/opticks/extg4/X4Material.cc:161
    #9  0x00007ffff7b63f34 in X4Material::X4Material (this=0x7fffffffc760, material=0x6d3db0, mode=71 'G') at /home/simon/opticks/extg4/X4Material.cc:114
    #10 0x00007ffff7b63e6c in X4Material::Convert (material=0x6d3db0, mode=71 'G') at /home/simon/opticks/extg4/X4Material.cc:89
    #11 0x0000000000402b27 in main (argc=1, argv=0x7fffffffc8f8) at /home/simon/opticks/extg4/tests/X4MaterialTest.cc:40
    (gdb) 

    (gdb) f 11
    #11 0x0000000000402b27 in main (argc=1, argv=0x7fffffffc8f8) at /home/simon/opticks/extg4/tests/X4MaterialTest.cc:40
    40	    GMaterial* wine = X4Material::Convert(water, mode_g4_interpolated_onto_domain ) ; 
    (gdb) f 10
    #10 0x00007ffff7b63e6c in X4Material::Convert (material=0x6d3db0, mode=71 'G') at /home/simon/opticks/extg4/X4Material.cc:89
    89	    X4Material xmat(material, mode);
    (gdb) f 9
    #9  0x00007ffff7b63f34 in X4Material::X4Material (this=0x7fffffffc760, material=0x6d3db0, mode=71 'G') at /home/simon/opticks/extg4/X4Material.cc:114
    114	    init() ;
    (gdb) f 8
    #8  0x00007ffff7b6421a in X4Material::init (this=0x7fffffffc760) at /home/simon/opticks/extg4/X4Material.cc:161
    161	        X4MaterialPropertiesTable::Convert( m_mat, m_mpt, m_mode );
    (gdb) f 7
    #7  0x00007ffff7b68b6a in X4MaterialPropertiesTable::Convert (pmap=0x6d9b40, mpt=0x6d41e0, mode=71 'G') at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:40
    40	    X4MaterialPropertiesTable xtab(pmap, mpt, mode);
    (gdb) f 6
    #6  0x00007ffff7b68bdb in X4MaterialPropertiesTable::X4MaterialPropertiesTable (this=0x7fffffffc4e0, pmap=0x6d9b40, mpt=0x6d41e0, mode=71 'G')
        at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:49
    49	    init();
    (gdb) f 5
    #5  0x00007ffff7b68c0f in X4MaterialPropertiesTable::init (this=0x7fffffffc4e0) at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:54
    54	    AddProperties( m_pmap, m_mpt, m_mode );    
    (gdb) f 4
    #4  0x00007ffff7b693e4 in X4MaterialPropertiesTable::AddProperties (pmap=0x6d9b40, mpt=0x6d41e0, mode=71 'G') at /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:155
    155	        assert( pidx > -1 );  
    (gdb) p pidx
    $1 = -1
    (gdb) 








::


    140     G4double GetLowEdgeEnergy(size_t binNumber) const;
    141          // Obsolete method
    142          // Get the energy value at the low edge of the specified bin.
    143          // Take note that the 'binNumber' starts from '0'.
    144          // The boundary check will not be done.


    060 inline
     61  G4double G4PhysicsVector::Energy(const size_t index) const
     62 {
     63   return binVector[index];
     64 }
     65 

    151 G4double G4PhysicsVector::GetLowEdgeEnergy(size_t binNumber) const
    152 {
    153   return binVector[binNumber];
    154 }
    155 

    130 inline
    131 G4double G4PhysicsOrderedFreeVector::GetMinLowEdgeEnergy()
    132 {
    133   return binVector.front();
    134 }







epsilon:tmp blyth$ opticks-fl GetMinLowEdgeEnergy
./cfg4/C4Cerenkov1042.cc
./cfg4/DsG4Cerenkov.cc
./cfg4/CMaterialLib.cc
./cfg4/G4Cerenkov1042.cc
./cfg4/CMPT.cc
./cfg4/OpRayleigh.cc
./cfg4/CCerenkovGenerator.cc
./cfg4/Cerenkov.cc
./extg4/tests/X4ScintillationTest.cc
./extg4/tests/X4ArrayTest.cc
./extg4/X4MaterialPropertyVector.cc
./extg4/X4MaterialPropertyVector.hh
./qudarap/qsim.h
./examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc
./examples/Geant4/CerenkovStandalone/L4CerenkovTest.cc
./examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc


epsilon:opticks blyth$ opticks-fl GetMaxLowEdgeEnergy
./cfg4/C4Cerenkov1042.cc
./cfg4/DsG4Cerenkov.cc
./cfg4/CMaterialLib.cc
./cfg4/G4Cerenkov1042.cc
./cfg4/CMPT.cc
./cfg4/OpRayleigh.cc
./cfg4/CCerenkovGenerator.cc
./cfg4/Cerenkov.cc

./extg4/tests/X4ScintillationTest.cc
./extg4/tests/X4ArrayTest.cc
./extg4/X4MaterialPropertyVector.cc
./extg4/X4MaterialPropertyVector.hh

./qudarap/qsim.h
./examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc
./examples/Geant4/CerenkovStandalone/L4CerenkovTest.cc
./examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc
epsilon:opticks blyth$ 


epsilon:opticks blyth$ opticks-fl IsFilledVectorExist 
./cfg4/C4Cerenkov1042.cc
./cfg4/DsG4Cerenkov.cc
./cfg4/G4Cerenkov1042.cc
./cfg4/Cerenkov.cc
./extg4/X4MaterialPropertyVector.cc
./extg4/X4MaterialPropertyVector.hh
./examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc
./examples/Geant4/CerenkovStandalone/L4CerenkovTest.cc
./examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc
epsilon:opticks blyth$ 



epsilon:opticks blyth$ opticks-f GetMinLowEdgeEnergy
./cfg4/C4Cerenkov1042.cc:  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./cfg4/C4Cerenkov1042.cc:  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./cfg4/DsG4Cerenkov.cc:	G4double Pmin = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMinLowEdgeEnergy();
./cfg4/DsG4Cerenkov.cc:	G4double Pmin = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMinLowEdgeEnergy();
./cfg4/CMaterialLib.cc:        G4double Pmin = rindex->GetMinLowEdgeEnergy();
./cfg4/G4Cerenkov1042.cc:  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./cfg4/G4Cerenkov1042.cc:  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./cfg4/CMPT.cc:                  << " MinLowEdgeEnergy " << v->GetMinLowEdgeEnergy()
./cfg4/OpRayleigh.cc:               << " fdom(Min) " << std::setw(15) << std::fixed << std::setprecision(3) << rayleigh->GetMinLowEdgeEnergy()
./cfg4/CCerenkovGenerator.cc:    G4double Pmin2 = Rindex->GetMinLowEdgeEnergy();
./cfg4/Cerenkov.cc:	G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./cfg4/Cerenkov.cc:	G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./extg4/tests/X4ScintillationTest.cc:    double e1 = ScintillatorIntegral->GetMinLowEdgeEnergy();
./extg4/tests/X4ArrayTest.cc:        << std::setw(30) << "GetMinLowEdgeEnergy() " 
./extg4/tests/X4ArrayTest.cc:        << std::fixed << std::setw(10) << std::setprecision(5) << vec->GetMinLowEdgeEnergy() 
./extg4/X4MaterialPropertyVector.cc:G4double X4MaterialPropertyVector::GetMinLowEdgeEnergy( const G4MaterialPropertyVector* mpv ) // static 
./extg4/X4MaterialPropertyVector.cc:    return const_cast<G4MaterialPropertyVector*>(mpv)->GetMinLowEdgeEnergy(); 
./extg4/X4MaterialPropertyVector.cc:G4double X4MaterialPropertyVector::GetMinLowEdgeEnergy() const
./extg4/X4MaterialPropertyVector.cc:    return GetMinLowEdgeEnergy(vec); 
./extg4/X4MaterialPropertyVector.hh:    static G4double GetMinLowEdgeEnergy( const G4MaterialPropertyVector* vec ); 
./extg4/X4MaterialPropertyVector.hh:    G4double GetMinLowEdgeEnergy() const ; 
./qudarap/qsim.h:    251   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc:	G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc:	G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./examples/Geant4/CerenkovStandalone/L4CerenkovTest.cc:	G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./examples/Geant4/CerenkovStandalone/L4CerenkovTest.cc:    G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc:  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
./examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc:  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
epsilon:opticks blyth$ 






    060 inline
     61  G4double G4PhysicsVector::Energy(const size_t index) const
     62 {
     63   return binVector[index];
     64 }
     65 
     66 //---------------------------------------------------------------
     67 
     68 inline
     69  G4double G4PhysicsVector::GetMaxEnergy() const
     70 {
     71   return edgeMax;
     72 }
     73 


    112 inline
    113 G4double G4PhysicsOrderedFreeVector::GetMaxValue()
    114 {
    115   return dataVector.back();
    116 }
    117 
    118 inline
    119 G4double G4PhysicsOrderedFreeVector::GetMinValue()
    120 {
    121   return dataVector.front();
    122 }
    123 
    124 inline
    125 G4double G4PhysicsOrderedFreeVector::GetMaxLowEdgeEnergy()
    126 {
    127   return binVector.back();
    128 }
    129 
    130 inline
    131 G4double G4PhysicsOrderedFreeVector::GetMinLowEdgeEnergy()
    132 {
    133   return binVector.front();
    134 }


