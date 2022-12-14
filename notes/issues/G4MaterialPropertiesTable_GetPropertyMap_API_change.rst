G4MaterialPropertiesTable_GetPropertyMap_API_change
=====================================================


Hans::

    Just wanted to let you know that with Geant4 11 there is an API change the way
    the material properties are accessed the ->GetPropertyMap();  method is gone so
    to make the code work with newer version of Geant4 one has ti encapsulate that
    piece of code and replace it with new accessors (like in the attached file).  I
    will see that I go through the code and change it were necessary. 


    /data3/wenzel/newopticks_dev5/opticks/u4/U4MaterialPropertiesTable.h: In static member function ‘static std::string U4MaterialPropertiesTable::DescPropertyMap(const G4MaterialPropertiesTable*)’:
    /data3/wenzel/newopticks_dev5/opticks/u4/U4MaterialPropertiesTable.h:37:27: error: ‘const class G4MaterialPropertiesTable’ has no member named ‘GetPropertyMap’; did you mean ‘GetProperty’?
       37 |     const MIV* miv = mpt->GetPropertyMap();
          |                           ^~~~~~~~~~~~~~
          |                           GetProperty





::

    epsilon:opticks blyth$ opticks-f GetPropertyMap
    ./extg4/X4MaterialPropertiesTable.cc:    auto m = mpt->GetPropertyMap();    // map is created at every call in 1100 
    ./extg4/X4MaterialPropertiesTable.cc:    auto m = mpt->GetPropertyMap(); 
    ./u4/U4Material.cc:    const MIV* miv =  mpt->GetPropertyMap(); 
    ./u4/U4Material.cc:    ss << " GetPropertyMap " << std::endl ; 
    ./u4/U4Material.cc:    const MIV* miv =  mpt->GetPropertyMap(); 
    ./u4/U4MaterialPropertiesTable.h:    const MIV* miv = mpt->GetPropertyMap(); 

    epsilon:opticks blyth$ 


* all use of GetPropertyMap in X4MaterialPropertiesTable.cc U4Material.cc are within "G4VERSION_NUMBER < 1100"
* there is one bare usage to fix in U4MaterialPropertiesTable.h


::

    epsilon:opticks blyth$ find . -name G4MaterialPropertiesTableTest.cc
    ./cfg4/tests/G4MaterialPropertiesTableTest.cc





These changes look mostly like my changes, not Hans changes to cope with lack of GetPropertyMap ?::

    epsilon:opticks blyth$ diff cfg4/tests/G4MaterialPropertiesTableTest.cc ~/Downloads/G4MaterialPropertiesTableTest.cc 
    23d22
    < #include "G4Version.hh"
    25d23
    < #include "X4MaterialPropertiesTable.hh"
    32a31
    >     G4bool warning ; 
    38c37
    <         G4int idx = mpt->GetPropertyIndex(pn); 
    ---
    >         G4int idx = mpt->GetPropertyIndex(pn, warning=true); 
    40c39
    <         MPV* mpv = mpt->GetProperty(idx); 
    ---
    >         MPV* mpv = mpt->GetProperty(idx, warning=false ); 
    56a56
    >     G4bool warning ; 
    65c65
    <         G4int idx = mpt->GetConstPropertyIndex(pn); 
    ---
    >         G4int idx = mpt->GetConstPropertyIndex(pn, warning=true); 
    85,100d84
    < void test_GetProperty_NonExisting(const G4MaterialPropertiesTable* mpt_)
    < {
    <     G4MaterialPropertiesTable* mpt = const_cast<G4MaterialPropertiesTable*>(mpt_);   // tut tut GetProperty is not const correct 
    < 
    <     const char* key = "NonExistingKey" ; 
    < #if G4VERSION_NUMBER < 1100 
    <     G4MaterialPropertyVector* mpv = mpt->GetProperty(key); 
    < #else
    <     G4MaterialPropertyVector* mpv = X4MaterialPropertiesTable::GetProperty(mpt, key); 
    < #endif
    < 
    <     LOG(info) << " key " << key << " mpv " << mpv ; 
    <     assert( mpv == nullptr ); 
    < }
    < 
    < 
    104c88
    <     OPTICKS_LOG(argc, argv);
    ---
    >     OPTICKS_LOG__(argc, argv);
    124,125d107
    <     test_GetProperty_NonExisting(mpt); 
    < 
    epsilon:opticks blyth$ 



::

    epsilon:opticks blyth$ opticks-f GetPropertyMap
    ./extg4/X4MaterialPropertiesTable.cc:    auto m = mpt->GetPropertyMap();    // map is created at every call in 1100 
    ./extg4/X4MaterialPropertiesTable.cc:    auto m = mpt->GetPropertyMap(); 
    ./u4/U4MaterialPropertiesTable.h:    const MIV* miv = mpt->GetPropertyMap(); 
    ./u4/U4Material.cc:    const MIV* miv =  mpt->GetPropertyMap(); 
    ./u4/U4Material.cc:    ss << " GetPropertyMap " << std::endl ; 
    ./u4/U4Material.cc:    const MIV* miv =  mpt->GetPropertyMap(); 
    epsilon:opticks blyth$ 



