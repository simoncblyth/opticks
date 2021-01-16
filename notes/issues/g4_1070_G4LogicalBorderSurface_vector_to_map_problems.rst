g4_1070_G4LogicalBorderSurface_vector_to_map_problems
=========================================================


::

    === om-make-one : extg4           /Users/francis/opticks/extg4                                 /Users/francis/local/opticks/build/extg4                     
    Scanning dependencies of target ExtG4
    [  2%] Building CXX object CMakeFiles/ExtG4.dir/X4Gen.cc.o
    [  2%] Building CXX object CMakeFiles/ExtG4.dir/X4CSG.cc.o
    [  3%] Building CXX object CMakeFiles/ExtG4.dir/X4.cc.o
    /Users/francis/opticks/extg4/X4.cc:323:25: error: no matching function for call to 'GetItemIndex'
        int idx_lbs = lbs ? GetItemIndex<G4LogicalBorderSurface>( lbs_table, lbs ) : -1 ;    
                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /Users/francis/opticks/extg4/X4.cc:111:9: note: candidate function not viable: no known conversion from 'const G4LogicalBorderSurfaceTable *' (aka 'const map<std::pair<const G4VPhysicalVolume *, const
          G4VPhysicalVolume *>, G4LogicalBorderSurface *> *') to 'const std::vector<G4LogicalBorderSurface *> *' for 1st argument
    int X4::GetItemIndex( const std::vector<T*>* vec, const T* const item )
            ^
    1 error generated.
    make[2]: *** [CMakeFiles/ExtG4.dir/X4.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....



It would be simple to assign a creation order index to the G4LogicalBorderSurface::

     44 G4LogicalBorderSurface::
     45 G4LogicalBorderSurface(const G4String& name,
     46                              G4VPhysicalVolume* vol1,
     47                              G4VPhysicalVolume* vol2,
     48                              G4SurfaceProperty* surfaceProperty)
     49   : G4LogicalSurface(name, surfaceProperty),
     50     Volume1(vol1), Volume2(vol2), 
            Index( theBorderSurfaceTable ? theBorderSurfaceTable->size() : 0 )  // Assign creation order index to the border surface 
     51 {
     52   if (theBorderSurfaceTable == nullptr)
     53   {
     54     theBorderSurfaceTable = new G4LogicalBorderSurfaceTable;
     55   }
     56 
     57   // Store in the table of Surfaces
     58   //
     59   theBorderSurfaceTable->insert(std::make_pair(std::make_pair(vol1,vol2),this));
     60 }
     61 


With the creation order index can explicitly control the ordering despite using a std::map with std::pair of pointers key::

    size_t GetIndex() const ; 


    inline
    size_t G4LogicalBorderSurface::GetIndex() const 
    {
      return Index;
    }
       


     


