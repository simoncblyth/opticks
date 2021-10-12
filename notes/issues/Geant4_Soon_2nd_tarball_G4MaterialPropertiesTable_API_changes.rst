Geant4_Soon_2nd_tarball_G4MaterialPropertiesTable_API_changes
===============================================================


Hi Soon, Daren, 

Between the tarballs there are API+behaviour changes to G4MaterialPropertiesTable 
that make attempts to access a non-existing property/const-property 
throw fatal exceptions.

As I pointed out previously this change is not appropriate as it removes 
the ability to introspect what properties exist in the table.  

Given that the API allows the addition and removal of properties/const-properties
removing introspection is removing an essential feature. 
Without introspection it becomes a CRUD(create-read-update-delete) interface where 
read is only usable when you externally keep track of what you have added and not removed. 

But surely the point of a properties table is to keep track of the properties ?

Introspection removal means that the ConstPropertyExists method is never able to return false 
because GetConstPropertyIndex will always throw fatal exceptions for non-existing keys.
A boolean accessor that can only ever return true is worse than useless. 

    242 G4bool G4MaterialPropertiesTable::ConstPropertyExists(const G4String& key) const
    243 {
    244   // Returns true if a const property 'key' exists
    245   return ConstPropertyExists(GetConstPropertyIndex(key));
    246 }

    171 G4int G4MaterialPropertiesTable::GetConstPropertyIndex(
    172   const G4String& key) const
    173 {
    174   // Returns the constant material property index corresponding to a key
    175 
    176   size_t index = std::distance(
    177     fMatConstPropNames.begin(),
    178     std::find(fMatConstPropNames.begin(), fMatConstPropNames.end(), key));
    179   if(index < fMatConstPropNames.size())
    180     return index;
    181 
    182   G4ExceptionDescription ed;
    183   ed << "Constant Material Property Index for key " << key << " not found.";
    184   G4Exception("G4MaterialPropertiesTable::GetConstPropertyIndex()", "mat200",
    185               FatalException, ed);
    186   return 0;
    187 }

Similarly the Get methods have all changed behaviour to throw fatal exceptions, where 
previously they returned nullptr for example.

In order to make the API complete again it is necessary to fix ConstPropertyExists
and add a new method PropertyExists that is able to return both true and false without 
throwing exceptions.  

An alternative to adding PropertyExists is for the GetProperty method to regain the prior 
functionality of simply returning nullptr when the property does not exist. 
Note that a key existing and are key+property existing are distinct. 

An example of a situation where introspection is useful is with 
code that works with multiple geometries (perhaps loaded from GDML) 
in which a variety of possible custom properties may be present 
and which needs to branch depending on which properties are there.
For example the presence (or absence) of some properties might indicate that 
some more properties need to be added or that some pre-processing is needed 
to be done, such as creating lookup tables prior to simulation. 

I am sure that over the past 10 years or more the many users of G4MaterialPropertiesTable 
have come up with many other uses for introspection so removing this important feature 
will cost them significant effort if they wish to keep their code operational.

Fortunately the map accessors GetPropertyMap/GetConstPropertyMap were not removed.
They allowed me to workaround the feature loss with static methods in X4MaterialPropertiesTable 
which have enabled me to get all the Opticks tests to pass with the 2nd tarball. 

1042::

    SLOW: tests taking longer that 15 seconds

    FAILS:  0   / 501   :  Sat Oct  9 04:49:48 2021   
    O[blyth@localhost opticks]$ 


1100, Soon 2nd tarball geant4.master100721.tar::

    SLOW: tests taking longer that 15 seconds
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         35.47  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         35.54  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Passed                         35.34  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Passed                         37.76  

    FAILS:  0   / 501   :  Sat Oct  9 04:57:42 2021   
    (base) [simon@localhost opticks]$ 



Simon








x4 compilation fails due to API changes::

    [ 38%] Building CXX object CMakeFiles/ExtG4.dir/X4GDML.cc.o
    [ 40%] Building CXX object CMakeFiles/ExtG4.dir/X4_LOG.cc.o
    [ 40%] Building CXX object CMakeFiles/ExtG4.dir/X4GDMLMatrix.cc.o
    /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc: In static member function ‘static void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>*, const G4MaterialPropertiesTable*, char)’:
    /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:79:63: error: no matching function for call to ‘G4MaterialPropertiesTable::GetPropertyIndex(const string&, G4bool&) const’
             G4int pidx = mpt->GetPropertyIndex(pname, warning=true);
                                                                   ^
    In file included from /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:21:
    /data/simon/local/opticks_externals/g4_100072/include/Geant4/G4MaterialPropertiesTable.hh:130:9: note: candidate: ‘G4int G4MaterialPropertiesTable::GetPropertyIndex(const G4String&) const’
       G4int GetPropertyIndex(const G4String& key) const;
             ^~~~~~~~~~~~~~~~
    /data/simon/local/opticks_externals/g4_100072/include/Geant4/G4MaterialPropertiesTable.hh:130:9: note:   candidate expects 1 argument, 2 provided
    /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:81:98: error: no matching function for call to ‘G4MaterialPropertiesTable::GetProperty(G4int&, G4bool&)’
             MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );
                                                                                                      ^
    In file included from /home/simon/opticks/extg4/X4MaterialPropertiesTable.cc:21:
    /data/simon/local/opticks_externals/g4_100072/include/Geant4/G4MaterialPropertiesTable.hh:116:29: note: candidate: ‘G4MaterialPropertyVector* G4MaterialPropertiesTable::GetProperty(const char*) const’
       G4MaterialPropertyVector* GetProperty(const char* key) const;
                                 ^~~~~~~~~~~
    /data/simon/local/opticks_externals/g4_100072/include/Geant4/G4MaterialPropertiesTable.hh:116:29: note:   candidate expects 1 argument, 2 provided
    /data/simon/local/opticks_externals/g4_100072/include/Geant4/G4MaterialPropertiesTable.hh:117:29: note: candidate: ‘G4MaterialPropertyVector* G4MaterialPropertiesTable::GetProperty(const G4String&) const’
       G4MaterialPropertyVector* GetProperty(const G4String& key) const;
                                 ^~~~~~~~~~~



Compare Soon tarballs 0 and 1::

    (base) [simon@localhost opticks]$ cd /data/simon/local/opticks_externals
    (base) [simon@localhost opticks_externals]$ ln -s g4_91072.build/geant4.10.7.r08 0
    (base) [simon@localhost opticks_externals]$ ln -s g4_100072.build/geant4.master100721 1

    (base) [simon@localhost opticks_externals]$ diff {0,1}/source/materials/include/G4MaterialPropertiesTable.hh
    116,121c116,118
    <   G4MaterialPropertyVector* GetProperty(const char* key,
    <                                         G4bool warning = false) const;
    <   G4MaterialPropertyVector* GetProperty(const G4String& key,
    <                                         G4bool warning = false) const;
    <   G4MaterialPropertyVector* GetProperty(const G4int index,
    <                                         G4bool warning = false) const;
    ---
    >   G4MaterialPropertyVector* GetProperty(const char* key) const;
    >   G4MaterialPropertyVector* GetProperty(const G4String& key) const;
    >   G4MaterialPropertyVector* GetProperty(const G4int index) const;
    130,131c127
    <   G4int GetConstPropertyIndex(const G4String& key,
    <                               G4bool warning = false) const;
    ---
    >   G4int GetConstPropertyIndex(const G4String& key) const;
    134c130
    <   G4int GetPropertyIndex(const G4String& key, G4bool warning = false) const;
    ---
    >   G4int GetPropertyIndex(const G4String& key) const;
    (base) [simon@localhost opticks_externals]$ 



    (base) [simon@localhost opticks_externals]$ diff {0,1}/source/materials/src/G4MaterialPropertiesTable.cc
    171,172c171,172
    < G4int G4MaterialPropertiesTable::GetConstPropertyIndex(const G4String& key,
    <                                                        G4bool warning) const
    ---
    > G4int G4MaterialPropertiesTable::GetConstPropertyIndex(
    >   const G4String& key) const
    181,188c181,186
    <   if(warning)
    <   {
    <     G4ExceptionDescription ed;
    <     ed << "Constant Material Property Index for key " << key << " not found.";
    <     G4Exception("G4MaterialPropertiesTable::GetConstPropertyIndex()", "mat200",
    <                 JustWarning, ed);
    <   }
    <   return -1;
    ---
    > 
    >   G4ExceptionDescription ed;
    >   ed << "Constant Material Property Index for key " << key << " not found.";
    >   G4Exception("G4MaterialPropertiesTable::GetConstPropertyIndex()", "mat200",
    >               FatalException, ed);
    >   return 0;
    191,192c189
    < G4int G4MaterialPropertiesTable::GetPropertyIndex(const G4String& key,
    <                                                   G4bool warning) const
    ---
    > G4int G4MaterialPropertiesTable::GetPropertyIndex(const G4String& key) const
    200,207c197,201
    <   if(warning)
    <   {
    <     G4ExceptionDescription ed;
    <     ed << "Material Property Index for key " << key << " not found.";
    <     G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()", "mat201",
    <                 JustWarning, ed);
    <   }
    <   return -1;
    ---
    >   G4ExceptionDescription ed;
    >   ed << "Material Property Index for key " << key << " not found.";
    >   G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()", "mat201",
    >               FatalException, ed);
    >   return 0;
    260c254
    <   const G4String& key, G4bool warning) const
    ---
    >   const G4String& key) const
    263c257
    <   const G4int index = GetPropertyIndex(key, warning);
    ---
    >   const G4int index = GetPropertyIndex(key);
    268c262
    <   const char* key, G4bool warning) const
    ---
    >   const char* key) const
    270,271c264,265
    <   const G4int index = GetPropertyIndex(G4String(key), warning);
    <   return GetProperty(index, warning);
    ---
    >   const G4int index = GetPropertyIndex(G4String(key));
    >   return GetProperty(index);
    275c269
    <   const G4int index, G4bool warning) const
    ---
    >   const G4int index) const
    281,287c275,279
    <   if(warning)
    <   {
    <     G4ExceptionDescription ed;
    <     ed << "Material Property for index " << index << " not found.";
    <     G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()", "mat203",
    <                 JustWarning, ed);
    <   }
    ---
    > 
    >   G4ExceptionDescription ed;
    >   ed << "Material Property for index " << index << " not found.";
    >   G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()", "mat203",
    >               FatalException, ed);
    (base) [simon@localhost opticks_externals]$ 





Obnoxious behavior change, means that GetPropertyIndex is cannot be used for discovery::

    189 G4int G4MaterialPropertiesTable::GetPropertyIndex(const G4String& key) const
    190 {
    191   // Returns the material property index corresponding to a key
    192   size_t index =
    193     std::distance(fMatPropNames.begin(),
    194                   std::find(fMatPropNames.begin(), fMatPropNames.end(), key));
    195   if(index < fMatPropNames.size())
    196     return index;
    197   G4ExceptionDescription ed;
    198   ed << "Material Property Index for key " << key << " not found.";
    199   G4Exception("G4MaterialPropertiesTable::GetPropertyIndex()", "mat201",
    200               FatalException, ed);
    201   return 0;
    202 }



Following updates to compile with new API::


    SLOW: tests taking longer that 15 seconds
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         36.89  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         35.99  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Passed                         36.04  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Passed                         38.51  


    FAILS:  7   / 501   :  Sat Oct  9 00:30:34 2021   
      9  /36  Test #9  : ExtG4Test.X4MaterialTest                      Child aborted***Exception:     0.17   
      12 /36  Test #12 : ExtG4Test.X4MaterialTableTest                 Child aborted***Exception:     0.17   
      13 /36  Test #13 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.17   
      14 /36  Test #14 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.17   
      30 /36  Test #30 : ExtG4Test.X4MaterialPropertiesTableTest       Child aborted***Exception:     0.17   
      15 /45  Test #15 : CFG4Test.G4MaterialPropertiesTableTest        Child aborted***Exception:     0.24   
      16 /45  Test #16 : CFG4Test.CMPTTest                             Child aborted***Exception:     0.26   
    (base) [simon@localhost opticks]$ 


But old Geant4 1042 also, failing 2::

    FAILS:  2   / 501   :  Sat Oct  9 00:35:07 2021   
      13 /36  Test #13 : ExtG4Test.X4PhysicalVolumeTest                Subprocess aborted***Exception:   0.22   
      14 /36  Test #14 : ExtG4Test.X4PhysicalVolume2Test               Subprocess aborted***Exception:   0.14   
    O[blyth@localhost opticks]$ 




::

    Start  9: ExtG4Test.X4MaterialTest
     9/36 Test  #9: ExtG4Test.X4MaterialTest ................................Child aborted***Exception:   0.17 sec

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat200
          issued by : G4MaterialPropertiesTable::GetConstPropertyIndex()
    Constant Material Property Index for key EFFICIENCY not found.
    *** Fatal Exception ***
    -------- EEEE ------- G4Exception-END -------- EEEE -------


    *** G4Exception: Aborting execution ***

          Start 10: ExtG4Test.X4MaterialWaterStandaloneTest
    10/36 Test #10: ExtG4Test.X4MaterialWaterStandaloneTest .................   Passed    0.10 sec
          Start 11: ExtG4Test.X4MaterialWaterTest
    11/36 Test #11: ExtG4Test.X4MaterialWaterTest ...........................   Passed    0.17 sec
          Start 12: ExtG4Test.X4MaterialTableTest
    12/36 Test #12: ExtG4Test.X4MaterialTableTest ...........................Child aborted***Exception:   0.17 sec
    2021-10-09 00:27:30.859 FATAL [361917] [Opticks::envkey@348]  --allownokey option prevents key checking : this is for debugging of geocache creation 
    2021-10-09 00:27:30.865 FATAL [361917] [OpticksResource::init@122]  CAUTION : are allowing no key 

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat200
          issued by : G4MaterialPropertiesTable::GetConstPropertyIndex()
    Constant Material Property Index for key EFFICIENCY not found.
    *** Fatal Exception ***
    -------- EEEE ------- G4Exception-END -------- EEEE -------


    *** G4Exception: Aborting execution ***

          Start 13: ExtG4Test.X4PhysicalVolumeTest
    13/36 Test #13: ExtG4Test.X4PhysicalVolumeTest ..........................Child aborted***Exception:   0.17 sec

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat202
          issued by : G4MaterialPropertiesTable::GetConstProperty()
    Constant Material Property Index 0 not found.
    *** Fatal Exception ***
    -------- EEEE ------- G4Exception-END -------- EEEE -------


    *** G4Exception: Aborting execution ***

          Start 14: ExtG4Test.X4PhysicalVolume2Test
    14/36 Test #14: ExtG4Test.X4PhysicalVolume2Test .........................Child aborted***Exception:   0.17 sec

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat202
          issued by : G4MaterialPropertiesTable::GetConstProperty()
    Constant Material Property Index 0 not found.
    *** Fatal Exception ***
    -------- EEEE ------- G4Exception-END -------- EEEE -------





G4MaterialPropertiesTable::ConstPropertyExists throws exception for non-existing key::


    (gdb) bt
    #0  0x00007fffeafc4387 in raise () from /lib64/libc.so.6
    #1  0x00007fffeafc5a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffef2f5f7d in G4Exception (originOfException=0x7fffefbd21f0 "G4MaterialPropertiesTable::GetConstPropertyIndex()", exceptionCode=0x7fffefbd21e2 "mat200", severity=FatalException, 
        description=0x6dc5d8 "Constant Material Property Index for key EFFICIENCY not found.")
        at /data/simon/local/opticks_externals/g4_100072.build/geant4.master100721/source/global/management/src/G4Exception.cc:88
    #3  0x00007fffef2f614b in G4Exception (originOfException=0x7fffefbd21f0 "G4MaterialPropertiesTable::GetConstPropertyIndex()", exceptionCode=0x7fffefbd21e2 "mat200", severity=FatalException, 
        description=...) at /data/simon/local/opticks_externals/g4_100072.build/geant4.master100721/source/global/management/src/G4Exception.cc:104
    #4  0x00007fffefb63f95 in G4MaterialPropertiesTable::GetConstPropertyIndex (this=0x6d6c60, key=...)
        at /data/simon/local/opticks_externals/g4_100072.build/geant4.master100721/source/materials/src/G4MaterialPropertiesTable.cc:184
    #5  0x00007fffefb64419 in G4MaterialPropertiesTable::ConstPropertyExists (this=0x6d6c60, key=0x7ffff7ba05a5 "EFFICIENCY")
        at /data/simon/local/opticks_externals/g4_100072.build/geant4.master100721/source/materials/src/G4MaterialPropertiesTable.cc:250
    #6  0x00007ffff7b5f9f7 in X4Material::HasEfficiencyProperty (mpt_=0x6d6c60) at /home/simon/opticks/extg4/X4Material.cc:130
    #7  0x00007ffff7b5f899 in X4Material::X4Material (this=0x7fffffffc280, material=0x6d6830, mode=71 'G') at /home/simon/opticks/extg4/X4Material.cc:111
    #8  0x00007ffff7b5f800 in X4Material::Convert (material=0x6d6830, mode=71 'G') at /home/simon/opticks/extg4/X4Material.cc:89
    #9  0x00000000004044f0 in main (argc=1, argv=0x7fffffffc418) at /home/simon/opticks/extg4/tests/X4MaterialTest.cc:52
    (gdb) 




G4MaterialPropertiesTable::ConstPropertyExists can never return false::

    242 G4bool G4MaterialPropertiesTable::ConstPropertyExists(const G4String& key) const
    243 { 
    244   // Returns true if a const property 'key' exists
    245   return ConstPropertyExists(GetConstPropertyIndex(key));
    246 } 

    171 G4int G4MaterialPropertiesTable::GetConstPropertyIndex(
    172   const G4String& key) const
    173 {
    174   // Returns the constant material property index corresponding to a key
    175 
    176   size_t index = std::distance(
    177     fMatConstPropNames.begin(),
    178     std::find(fMatConstPropNames.begin(), fMatConstPropNames.end(), key));
    179   if(index < fMatConstPropNames.size())
    180     return index;
    181 
    182   G4ExceptionDescription ed;
    183   ed << "Constant Material Property Index for key " << key << " not found.";
    184   G4Exception("G4MaterialPropertiesTable::GetConstPropertyIndex()", "mat200",
    185               FatalException, ed);
    186   return 0;
    187 }



Now down to 3::

    FAILS:  3   / 501   :  Sat Oct  9 03:18:14 2021   
      30 /36  Test #30 : ExtG4Test.X4MaterialPropertiesTableTest       Child aborted***Exception:     0.16   
      15 /45  Test #15 : CFG4Test.G4MaterialPropertiesTableTest        Child aborted***Exception:     0.26   
      16 /45  Test #16 : CFG4Test.CMPTTest                             Child aborted***Exception:     0.27   
    (base) [simon@localhost opticks]$ 




* https://bitbucket.org/simoncblyth/opticks/commits/b51aa2eb441aacb334b1790d839e9513ecb31dbb

* https://bitbucket.org/simoncblyth/opticks/commits/e9b8c97d90efeae70fd9c6b0d6dccd1cd5f0624c

* https://bitbucket.org/simoncblyth/opticks/commits/2c9ec1679edb5b9a6df6126adfc95a913dcc495e

* https://bitbucket.org/simoncblyth/opticks/commits/f1c3b0fe5d77de00f2975ba896fc556d22c691c7

* https://bitbucket.org/simoncblyth/opticks/commits/9aaa6708ceae8080a648c8681c0f05bdfca93207

* https://bitbucket.org/simoncblyth/opticks/commits/f0ffedc8b6e8b817f85ab0130e1fd157c5726203



::

    (base) [simon@localhost opticks]$ git l -n6
    commit f0ffedc8b6e8b817f85ab0130e1fd157c5726203
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Oct 8 21:44:50 2021 +0100

        down to 0/501 fails with 1100, probably

    M       cfg4/tests/G4MaterialPropertiesTableTest.cc
    M       extg4/X4MaterialPropertiesTable.cc
    M       extg4/X4MaterialPropertiesTable.hh
    M       notes/issues/Geant4_Soon_2nd_tarball_G4MaterialPropertiesTable_API_changes.rst

    commit 9aaa6708ceae8080a648c8681c0f05bdfca93207
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Oct 8 21:14:00 2021 +0100

        workaround changed behavior of G4MaterialPropertiesTable::GetProperty with non-existing key, in 1100 it now throws exceptions : previously it returned null,  with X4MaterialPropertiesTable::GetPro

    M       cfg4/CMPT.cc
    M       cfg4/tests/G4MaterialPropertiesTableTest.cc
    M       extg4/X4MaterialPropertiesTable.cc
    M       extg4/X4MaterialPropertiesTable.hh

    commit f1c3b0fe5d77de00f2975ba896fc556d22c691c7
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Oct 8 20:40:28 2021 +0100

        avoid three more non-existing key exceptions in 1100 using X4MaterialPropertiesTable static workarounds

    M       cfg4/CMPT.cc
    M       extg4/X4MaterialPropertiesTable.cc
    M       extg4/tests/X4MaterialPropertiesTableTest.cc
    M       notes/issues/Geant4_Soon_2nd_tarball_G4MaterialPropertiesTable_API_changes.rst

    commit 2c9ec1679edb5b9a6df6126adfc95a913dcc495e
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Oct 8 19:52:30 2021 +0100

        more X4MaterialPropertiesTable static methods to workaround 1100 bugs

    M       cmake/Modules/OpticksCXXFlags.cmake
    M       extg4/X4Material.cc
    M       extg4/X4MaterialPropertiesTable.cc
    M       extg4/X4MaterialPropertiesTable.hh
    M       extg4/tests/X4MaterialPropertiesTableTest.cc
    M       notes/issues/Geant4_Soon_2nd_tarball_G4MaterialPropertiesTable_API_changes.rst

    commit e9b8c97d90efeae70fd9c6b0d6dccd1cd5f0624c
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Oct 8 17:22:33 2021 +0100

        updates for yet more G4MaterialPropertiesTable 1100 API changes, no more warning bool + no more index discovery

    M       cfg4/CDump.cc
    M       cfg4/CMPT.cc
    M       cfg4/CPropLib.cc
    M       cfg4/tests/CGDMLPropertyTest.cc
    M       cfg4/tests/G4MaterialPropertiesTableTest.cc
    M       extg4/X4Dump.cc
    M       extg4/X4MaterialPropertiesTable.cc
    M       extg4/X4MaterialPropertiesTable.hh
    M       extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
    M       extg4/tests/PhysicsFreeVectorTest.cc
    M       extg4/tests/X4MaterialPropertiesTableTest.cc

    commit b51aa2eb441aacb334b1790d839e9513ecb31dbb
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Oct 8 16:17:26 2021 +0100

        X4MaterialPropertiesTable::GetPropertyIndex workaround removal of property discovery functionality in G4 1100

    M       externals/g4.bash
    M       extg4/X4MaterialPropertiesTable.cc
    M       extg4/X4MaterialPropertiesTable.hh
    M       extg4/tests/CMakeLists.txt
    A       extg4/tests/X4MaterialPropertiesTableTest.cc
    A       notes/issues/Geant4_Soon_2nd_tarball_G4MaterialPropertiesTable_API_changes.rst







1042::

    SLOW: tests taking longer that 15 seconds

    FAILS:  0   / 501   :  Sat Oct  9 04:49:48 2021   
    O[blyth@localhost opticks]$ 


1100, Soon 2nd tarball::

    SLOW: tests taking longer that 15 seconds
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         35.47  
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         35.54  
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Passed                         35.34  
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Passed                         37.76  

    FAILS:  0   / 501   :  Sat Oct  9 04:57:42 2021   
    (base) [simon@localhost opticks]$ 





