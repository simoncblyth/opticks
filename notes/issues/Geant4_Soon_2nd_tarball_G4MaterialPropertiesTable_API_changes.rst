Geant4_Soon_2nd_tarball_G4MaterialPropertiesTable_API_changes
===============================================================


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


