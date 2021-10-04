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




1042::

    153 G4int G4MaterialPropertiesTable::GetConstPropertyIndex(const G4String& key,
    154                                                        G4bool warning) const
    155 {
    156   // Returns the constant material property index corresponding to a key
    157 
    158   size_t index = std::distance(G4MaterialConstPropertyName.begin(),
    159                      std::find(G4MaterialConstPropertyName.begin(),
    160                                      G4MaterialConstPropertyName.end(), key));
    161   if(index < G4MaterialConstPropertyName.size()) return index;
    162   if (warning) {
    163     G4ExceptionDescription ed;
    164     ed << "Constant Material Property Index for key " << key << " not found.";
    165     G4Exception("G4MaterialPropertiesTable::GetConstPropertyIndex()","mat206",
    166                 JustWarning, ed);
    167   }
    168   return -1;
    169 }


91072::

    171 G4int G4MaterialPropertiesTable::GetConstPropertyIndex(const G4String& key,
    172                                                        G4bool warning) const
    173 {
    174   // Returns the constant material property index corresponding to a key
    175 
    176   size_t index = std::distance(
    177     fMatConstPropNames.begin(),
    178     std::find(fMatConstPropNames.begin(), fMatConstPropNames.end(), key));
    179   if(index < fMatConstPropNames.size())
    180     return index;
    181   if(warning)
    182   {
    183     G4ExceptionDescription ed;
    184     ed << "Constant Material Property Index for key " << key << " not found.";
    185     G4Exception("G4MaterialPropertiesTable::GetConstPropertyIndex()", "mat200",
    186                 JustWarning, ed);
    187   }
    188   return -1;
    189 }




Old behavior::


    epsilon:extg4 blyth$ X4MaterialPropertiesTable=INFO X4MaterialTest 
    PLOG::EnvLevel adjusting loglevel by envvar   key X4MaterialPropertiesTable level INFO fallback DEBUG
    2021-10-02 20:00:16.598 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@71]  MaterialPropertyNames pns.size 23
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                         RINDEX pidx :     0 pvec :   0x7fab99d1c830
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                     EFFICIENCY pidx :     4 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                       GROUPVEL pidx :     9 pvec :   0x7fab99d1ce00
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                          MIEHG pidx :    10 pvec :   0x7fab99d1da60
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2021-10-02 20:00:16.599 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                      ABSLENGTH pidx :    14 pvec :   0x7fab99d1ce90
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  FASTCOMPONENT pidx :    15 pvec :   0x7fab99d1d480
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  SLOWCOMPONENT pidx :    16 pvec :   0x7fab99d1d710
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@82]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@135]  pns 23 pns_null 17
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :                  MIEHG_FORWARD pidx :     5 pval :             0.99
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :                 MIEHG_BACKWARD pidx :     6 pval :             0.99
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :            MIEHG_FORWARD_RATIO pidx :     7 pval :              0.8
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :             SCINTILLATIONYIELD pidx :     8 pval :               50
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :                RESOLUTIONSCALE pidx :     9 pval :                1
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :               FASTTIMECONSTANT pidx :    10 pval :                1
    2021-10-02 20:00:16.600 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :               SLOWTIMECONSTANT pidx :    12 pval :               10
    2021-10-02 20:00:16.601 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@161]  pname :                     YIELDRATIO pidx :    14 pval :              0.8
    2021-10-02 20:00:16.601 INFO  [14291299] [X4MaterialPropertiesTable::AddProperties@173]  cpns 33 cpns_null 25
    GMaterial::Summary material 1 a07b91ac42b48c542fcb22267597ef6b Water
    RINDEX : e5f2b7a6407555de81972d7bee759738 : 761 



91072 huh looks like some mixup between props and const-props::

    (base) [simon@localhost extg4]$ X4MaterialPropertiesTable=INFO X4MaterialTest
    PLOG::EnvLevel adjusting loglevel by envvar   key X4MaterialPropertiesTable level INFO fallback DEBUG
    2021-10-03 03:07:18.550 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@71]  MaterialPropertyNames pns.size 28
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                         RINDEX pidx :     0 pvec :        0x1837470
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                   REFLECTIVITY pidx :     1 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                     REALRINDEX pidx :     2 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                IMAGINARYRINDEX pidx :     3 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                     EFFICIENCY pidx :     4 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  TRANSMITTANCE pidx :     5 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                       GROUPVEL pidx :     9 pvec :        0x1837730
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                          MIEHG pidx :    10 pvec :        0x1837250
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                       RAYLEIGH pidx :    11 pvec :                0
    2021-10-03 03:07:18.551 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                   WLSCOMPONENT pidx :    12 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                   WLSABSLENGTH pidx :    13 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  WLSCOMPONENT2 pidx :    14 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  WLSABSLENGTH2 pidx :    15 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                      ABSLENGTH pidx :    16 pvec :        0x1837360
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :        SCINTILLATIONCOMPONENT1 pidx :    23 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :        SCINTILLATIONCOMPONENT2 pidx :    24 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :        SCINTILLATIONCOMPONENT3 pidx :    25 pvec :                0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  FASTCOMPONENT pidx :    26 pvec :        0x1835eb0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@82]  pname :                  SLOWCOMPONENT pidx :    27 pvec :        0x18381b0
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@135]  pns 28 pns_null 22
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                         RINDEX pidx :    -1 pval :               -1
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                   REFLECTIVITY pidx :    -1 pval :               -1
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                     REALRINDEX pidx :    -1 pval :               -1
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                IMAGINARYRINDEX pidx :    -1 pval :               -1
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                     EFFICIENCY pidx :    -1 pval :               -1
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                  TRANSMITTANCE pidx :    -1 pval :               -1
    2021-10-03 03:07:18.552 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :           SPECULARLOBECONSTANT pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :          SPECULARSPIKECONSTANT pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :            BACKSCATTERCONSTANT pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                       GROUPVEL pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                          MIEHG pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                       RAYLEIGH pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                   WLSCOMPONENT pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                   WLSABSLENGTH pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                  WLSCOMPONENT2 pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                  WLSABSLENGTH2 pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                      ABSLENGTH pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :       PROTONSCINTILLATIONYIELD pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :       TRITONSCINTILLATIONYIELD pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :        ALPHASCINTILLATIONYIELD pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :          IONSCINTILLATIONYIELD pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :        SCINTILLATIONCOMPONENT1 pidx :    -1 pval :               -1
    2021-10-03 03:07:18.553 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :        SCINTILLATIONCOMPONENT2 pidx :    -1 pval :               -1
    2021-10-03 03:07:18.554 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :        SCINTILLATIONCOMPONENT3 pidx :    -1 pval :               -1
    2021-10-03 03:07:18.554 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                  FASTCOMPONENT pidx :    -1 pval :               -1
    2021-10-03 03:07:18.554 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@162]  pname :                  SLOWCOMPONENT pidx :    -1 pval :               -1
    2021-10-03 03:07:18.554 INFO  [352166] [X4MaterialPropertiesTable::AddProperties@174]  cpns 28 cpns_null 0
    2021-10-03 03:07:18.554 FATAL [352166] [GPropertyMap<T>::getNumProperties@1026] GPropertyMap<T>::getNumProperties prop/keys mismatch  prop 28 keys 34
    X4MaterialTest: /home/simon/opticks/ggeo/GPropertyMap.cc:1032: unsigned int GPropertyMap<T>::getNumProperties() const [with T = double]: Assertion `m_prop.size() == m_keys.size()' failed.
    Aborted (core dumped)
    (base) [simon@localhost extg4]$ 


Old::

    523 std::vector<G4String> G4MaterialPropertiesTable::GetMaterialPropertyNames() const
    524 {
    525   return G4MaterialPropertyName;;
    526 }
    527 
    528 std::vector<G4String> G4MaterialPropertiesTable::GetMaterialConstPropertyNames() const
    529 {
    530   return G4MaterialConstPropertyName;
    531 }


New::

    140   // the next four methods are used in persistency/GDML:
    141   const std::vector<G4String> GetMaterialPropertyNames() const
    142   {
    143     return fMatPropNames;
    144   }
    145   const std::vector<G4String> GetMaterialConstPropertyNames() const
    146   {
    147     return fMatPropNames;
    148   }


Adhoc fixes on S
------------------

Fix the assumed typo wrt fMatConstPropNames::

    (base) [simon@localhost ~]$ g4-cls G4MaterialPropertiesTable 
    /data/simon/local/opticks_externals/g4_91072.build/geant4.10.7.r08
    vi -R source/materials/include/G4MaterialPropertiesTable.hh source/materials/src/G4MaterialPropertiesTable.cc
    2 files to edit
    (base) [simon@localhost ~]$ vi source/materials/include/G4MaterialPropertiesTable.hh source/materials/src/G4MaterialPropertiesTable.cc
    2 files to edit
    (base) [simon@localhost ~]$ cd /data/simon/local/opticks_externals/g4_91072.build/geant4.10.7.r08
    (base) [simon@localhost geant4.10.7.r08]$ vi source/materials/include/G4MaterialPropertiesTable.hh source/materials/src/G4MaterialPropertiesTable.cc


    140   // the next four methods are used in persistency/GDML:
    141   const std::vector<G4String> GetMaterialPropertyNames() const
    142   {
    143     return fMatPropNames;
    144   }
    145   const std::vector<G4String> GetMaterialConstPropertyNames() const
    146   {
    147     return fMatConstPropNames;
    148   }


Hmm the AddProperty createNewKey=true is already there. Must be some other problem with GDML?::

    (base) [simon@localhost src]$ grep AddProperty *.cc
    G4GDMLReadMaterials.cc:    matprop->AddProperty(Strip(name), propvect, true);
    G4GDMLReadSolids.cc:    matprop->AddProperty(Strip(name), propvect, true);
    (base) [simon@localhost src]$ 
    (base) [simon@localhost src]$ pwd
    /data/simon/local/opticks_externals/g4_91072.build/geant4.10.7.r08/source/persistency/gdml/src
    (base) [simon@localhost src]$ 

S hmm building after this change will take a long time as the bugfix is in the header. Another reason to minimize headers::

    g4-build

S opticks-tl::


    SLOW: tests taking longer that 15 seconds
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     37.12  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     36.97  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     37.01  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     37.90  


    FAILS:  12  / 497   :  Sun Oct  3 02:18:39 2021   
      9  /35  Test #9  : ExtG4Test.X4MaterialTest                      Child aborted***Exception:     0.16   
      12 /35  Test #12 : ExtG4Test.X4MaterialTableTest                 Child aborted***Exception:     0.18   
      ## maybe constprop change can fix these

      13 /35  Test #13 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.18   
      14 /35  Test #14 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.18   
      34 /35  Test #34 : ExtG4Test.X4SurfaceTest                       Child aborted***Exception:     0.47   
      ## ?

      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     37.12  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     36.97  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     37.01  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     37.90  
      ## change to X4PropertyMap.cc may fix these fails

      1  /45  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     3.76   
      15 /45  Test #15 : CFG4Test.G4MaterialPropertiesTableTest        Child aborted***Exception:     0.25   
      29 /45  Test #29 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     3.17   
      ## might be fixed by constprop bugfix


    (base) [simon@localhost ~]$ 



Following constprop and X4PropertyMap fixes down to 4 slow and 1 fail

    SLOW: tests taking longer that 15 seconds
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         36.00  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         35.63  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Passed                         35.82  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     37.78  


    FAILS:  1   / 497   :  Mon Oct  4 18:53:27 2021   
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     37.78  
    (base) [simon@localhost opticks]$ 
     


Contrast with standard on O (1042) : looks like factor 5 GDML slowdown::

      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         7.71   
      4  /45  Test #4  : CFG4Test.CGDMLTest                            Passed                         0.07   
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         7.44   
      6  /45  Test #6  : CFG4Test.CGDMLPropertyTest                    Passed                         0.07   
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Passed                         7.76   
      8  /45  Test #8  : CFG4Test.G4MaterialTest                       Passed                         0.07   


    SLOW: tests taking longer that 15 seconds


    FAILS:  0   / 497   :  Mon Oct  4 18:59:00 2021   
    O[blyth@localhost opticks]$ 




CInterpolationTest : SEGV from bad property address
-------------------------------------------------------

::

     
    12    PMT_3inch_photocathode_logsurf1          Photocathode_opsurf_3inch pv1 PMT_3inch_inner1_phys0x6f2c580 #0 pv2 PMT_3inch_body_phys0x6f2c500 #0
    13    PMT_3inch_photocathode_logsurf2          Photocathode_opsurf_3inch pv1 PMT_3inch_body_phys0x6f2c500 #0 pv2 PMT_3inch_inner1_phys0x6f2c580 #0
    14           UpperChimneyTyvekSurface    UpperChimneyTyvekOpticalSurface pv1 pUpperChimneyLS0x79eb640 #0 pv2 pUpperChimneyTyvek0x79eb7e0 #0

    2021-10-04 19:02:50.213 INFO  [410139] [main@116]  interpolate (control with option: --nointerpol) 1 name CInterpolationTest_interpol.npy tex 38,4,2,761,4 out 38,4,2,761,4
    2021-10-04 19:02:50.213 INFO  [410139] [main@132]  wlow 60 wstep 1 nl 761
    2021-10-04 19:02:50.213 INFO  [410139] [main@156]  nb 38
    2021-10-04 19:02:50.213 INFO  [410139] [main@166]  i   0 omat  18 osur 4294967295 isur 4294967295 imat  18

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7a96857 in G4PhysicsVector::Value (this=0x1b1, e=2.0664033072200039e-05) at /data/simon/local/opticks_externals/g4_91072/include/Geant4/G4PhysicsVector.icc:213
    213	  if(e > edgeMin && e < edgeMax)
    (gdb) bt
    #0  0x00007ffff7a96857 in G4PhysicsVector::Value (this=0x1b1, e=2.0664033072200039e-05) at /data/simon/local/opticks_externals/g4_91072/include/Geant4/G4PhysicsVector.icc:213
    #1  0x00007ffff7ae75a5 in CMPT::sample (this=0xd078c10, a=0xd0784c0, offset=3044, _keys=0x409241 "GROUPVEL,,, ", low=60, step=1, nstep=761) at /home/simon/opticks/cfg4/CMPT.cc:633
    #2  0x0000000000404d81 in main (argc=1, argv=0x7fffffffcf28) at /home/simon/opticks/cfg4/tests/CInterpolationTest.cc:196
    (gdb) 

    (gdb) bt
    #0  0x00007ffff7a96857 in G4PhysicsVector::Value (this=0x1b1, e=2.0664033072200039e-05) at /data/simon/local/opticks_externals/g4_91072/include/Geant4/G4PhysicsVector.icc:213
    #1  0x00007ffff7ae75a5 in CMPT::sample (this=0xd078c10, a=0xd0784c0, offset=3044, _keys=0x409241 "GROUPVEL,,, ", low=60, step=1, nstep=761) at /home/simon/opticks/cfg4/CMPT.cc:633
    #2  0x0000000000404d81 in main (argc=1, argv=0x7fffffffcf28) at /home/simon/opticks/cfg4/tests/CInterpolationTest.cc:196
    (gdb) f 2
    #2  0x0000000000404d81 in main (argc=1, argv=0x7fffffffcf28) at /home/simon/opticks/cfg4/tests/CInterpolationTest.cc:196
    196	            ompt->sample(out, om_offset, mkeys, wlow, wstep, nl );  
    (gdb) p om_offset
    $1 = 3044
    (gdb) p mkeys
    $2 = 0x409241 "GROUPVEL,,, "
    (gdb) p wlow
    $3 = 60
    (gdb) p wstep
    $4 = 1
    (gdb) p nl
    $5 = 761
    (gdb) f 1
    #1  0x00007ffff7ae75a5 in CMPT::sample (this=0xd078c10, a=0xd0784c0, offset=3044, _keys=0x409241 "GROUPVEL,,, ", low=60, step=1, nstep=761) at /home/simon/opticks/cfg4/CMPT.cc:633
    633	            G4double value = pofv ? pofv->Value( photonMomentum ) : 0.f ;
    (gdb) p pofv
    $6 = (G4MaterialPropertyVector *) 0x1b1
    (gdb) p *pofv
    Cannot access memory at address 0x1b1
    (gdb) p photonMomentum
    $7 = 2.0664033072200039e-05
    (gdb) 



::

    595 void CMPT::sample(NPY<double>* a, unsigned offset, const char* _keys, double low, double step, unsigned nstep )
    596 {
    597    // CAUTION: used by cfg4/tests/CInterpolationTest.cc 
    598 
    599     std::vector<std::string> keys ;
    600     BStr::split(keys, _keys, ',') ;
    601 
    602     unsigned nkey = keys.size();
    603 
    604     unsigned ndim = a->getDimensions() ;
    605     assert(ndim == 5);
    606     unsigned nl = a->getShape(3);
    607     unsigned nm_ = a->getShape(4);  // 4 corresponding to double4 of props used in tex 
    608 
    609     double* values = a->getValues() + offset ;
    610 
    611 
    612     assert( nl == nstep );
    613 
    614     if( nm_ != nkey )
    615     {
    616         LOG(fatal) << " unexpected _keys " << _keys
    617                    << " nkey " << nkey
    618                    << " nm_ " << nm_
    619                    << " a " << a->getShapeString()
    620                    ;
    621     }
    622     assert( nm_ == nkey );
    623 
    624     for(unsigned l=0 ; l < nl ; l++)
    625     {  
    626         G4double wavelength = (low + l*step)*CLHEP::nm ;
    627         G4double photonMomentum = h_Planck*c_light/wavelength ;
    628 
    629         for(unsigned m=0 ; m < nm_ ; m++)
    630         {
    631             const char* key = keys[m].c_str();
    632             G4MaterialPropertyVector* pofv = getVec(key);
    633             G4double value = pofv ? pofv->Value( photonMomentum ) : 0.f ;
    634             *(values + l*nm_ + m) = value ;
    635         }
    636     }  
    637 }



Hmm blank key beyond GROUPVEL::

    629	        for(unsigned m=0 ; m < nm_ ; m++)
    630	        {
    631	            const char* key = keys[m].c_str(); 
    632	            G4MaterialPropertyVector* pofv = getVec(key);
    633	            G4double value = pofv ? pofv->Value( photonMomentum ) : 0.f ;
    634	            *(values + l*nm_ + m) = value ;
    635	        }
    636	    }   
    637	}
    (gdb) p key
    $8 = 0xd078a08 ""

getVec should be giving null with blank key ?


    575 G4MaterialPropertyVector* CMPT::getVec(const char* key) const
    576 {
    577     G4MaterialPropertyVector* pofv = NULL ;
    578     G4MaterialPropertyVector* mpv = m_mpt->GetProperty(key);
    579     if(mpv) pofv = static_cast<G4MaterialPropertyVector*>(mpv);
    580     return pofv ;
    581 }


::

    259 G4MaterialPropertyVector* G4MaterialPropertiesTable::GetProperty(
    260   const G4String& key, G4bool warning) const
    261 {
    262   // Returns a Material Property Vector corresponding to a key
    263   const G4int index = GetPropertyIndex(key, warning);
    264   return GetProperty(index);
    265 }
    266 
    267 G4MaterialPropertyVector* G4MaterialPropertiesTable::GetProperty(
    268   const char* key, G4bool warning) const
    269 {
    270   const G4int index = GetPropertyIndex(G4String(key), warning);
    271   return GetProperty(index, warning);
    272 }


A non-found key in the above gives index -1 which is passed to the below giving "random" addess fMP[-1]:: 

    273 
    274 G4MaterialPropertyVector* G4MaterialPropertiesTable::GetProperty(
    275   const G4int index, G4bool warning) const
    276 {
    277   // Returns a Material Property Vector corresponding to an index
    278 
    279   if(index < (G4int) fMP.size())
    280     return fMP[index];
    281   if(warning)
    282   {
    283     G4ExceptionDescription ed;
    284     ed << "Material Property for index " << index << " not found.";
    285     G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()", "mat203",
    286                 JustWarning, ed);
    287   }
    288   return nullptr;
    289 }


91072::

    191 G4int G4MaterialPropertiesTable::GetPropertyIndex(const G4String& key,
    192                                                   G4bool warning) const
    193 {
    194   // Returns the material property index corresponding to a key
    195   size_t index =
    196     std::distance(fMatPropNames.begin(),
    197                   std::find(fMatPropNames.begin(), fMatPropNames.end(), key));
    198   if(index < fMatPropNames.size())
    199     return index;
    200   if(warning)
    201   {
    202     G4ExceptionDescription ed;
    203     ed << "Material Property Index for key " << key << " not found.";
    204     G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()", "mat201",
    205                 JustWarning, ed);
    206   }
    207   return -1;
    208 }



1042::

    231 G4MaterialPropertyVector*
    232 G4MaterialPropertiesTable::GetProperty(const G4int index, G4bool warning)
    233 {
    234   // Returns a Material Property Vector corresponding to an index
    235   MPiterator i;
    236   i = MP.find(index);
    237   if ( i != MP.end() ) return i->second;
    238   if (warning) {
    239     G4ExceptionDescription ed;
    240     ed << "Material Property for index " << index << " not found.";
    241     G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()","mat208",
    242                  JustWarning, ed);
    243   }
    244   return nullptr;
    245 }
    246 


Add a test to try to capture this in isolation::

     85 void test_GetProperty_NonExisting(const G4MaterialPropertiesTable* mpt_)
     86 {
     87     G4MaterialPropertiesTable* mpt = const_cast<G4MaterialPropertiesTable*>(mpt_);   // tut tut GetProperty is not const correct 
     88     const char* key = "NonExistingKey" ;
     89     G4bool warning = false ;
     90     G4MaterialPropertyVector* mpv = mpt->GetProperty(key, warning);
     91     LOG(info) << " key " << key << " mpv " << mpv ;
     92     assert( mpv == nullptr );
     93 }



