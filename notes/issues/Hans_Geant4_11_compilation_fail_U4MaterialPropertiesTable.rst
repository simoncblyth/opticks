Hans_Geant4_11_compilation_fail_U4MaterialPropertiesTable
=============================================================




Hi


when trying to build the current head version of opticks against Geant4 11.0. I get the following error messages. 


regards 

Hans 



G4MaterialPropertiesTable::GetPropertyMap no longer in Geant4  (Is this a new removal of API ?)
--------------------------------------------------------------------------------------------------

::

    epsilon:u4 blyth$ opticks-f GetPropertyMap
    ./extg4/X4MaterialPropertiesTable.cc:    auto m = mpt->GetPropertyMap();    // map is created at every call in 1100 
    ./extg4/X4MaterialPropertiesTable.cc:    auto m = mpt->GetPropertyMap(); 
    ./u4/U4MaterialPropertiesTable.h:    const MIV* miv = mpt->GetPropertyMap(); 
    ./u4/U4MaterialPropertiesTable.h:    const MIV* miv = mpt->GetPropertyMap();   // <-- METHOD REMOVED IN LATEST G4
    ./u4/U4Material.cc:    const MIV* miv =  mpt->GetPropertyMap(); 
    ./u4/U4Material.cc:    ss << " GetPropertyMap " << std::endl ; 
    ./u4/U4Material.cc:    const MIV* miv =  mpt->GetPropertyMap(); 




::

    150 /**
    151 U4MaterialPropertiesTable::GetProperties
    152 ------------------------------------------
    153 
    154 This aims to provide an API that does not change with Geant4 version. 
    155 
    156 **/
    157 
    158 inline void U4MaterialPropertiesTable::GetProperties(
    159       std::vector<std::string>& keys,
    160       std::vector<G4MaterialPropertyVector*>& props, const G4MaterialPropertiesTable* mpt )
    161 {
    162     std::vector<G4String> names = mpt->GetMaterialPropertyNames();
    163 
    164     typedef std::map<G4int, G4MaterialPropertyVector*> MIV ;
    165     const MIV* miv = mpt->GetPropertyMap();   // <-- METHOD REMOVED IN LATEST G4
    166 
    167     for(MIV::const_iterator iv=miv->begin() ; iv != miv->end() ; iv++)
    168     {
    169         G4int i = iv->first ;
    170         G4MaterialPropertyVector* v = iv->second ;
    171         const std::string& name = names[i] ;
    172 
    173         keys.push_back(name);
    174         props.push_back(v) ;
    175     }
    176     // WHY IS THIS MESS NECESSARY TO DO SUCH AN OBVIOUS THING ?
    177 }



::

    238 #if G4VERSION_NUMBER < 1100
    239 void U4Material::GetPropertyNames( std::vector<std::string>& names, const G4Material* mat )
    240 {
    241     G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    242     const G4MaterialPropertiesTable* mpt_ = mat->GetMaterialPropertiesTable();
    243 
    244     if( mpt_ == nullptr ) return ;
    245 
    246     std::vector<G4String> pnames = mpt_->GetMaterialPropertyNames();
    247 
    248     typedef std::map<G4int, G4MaterialPropertyVector*, std::less<G4int> > MIV ;
    249     const MIV* miv =  mpt->GetPropertyMap();
    250     for(MIV::const_iterator it=miv->begin() ; it != miv->end() ; it++ )
    251     {
    252          G4String name = pnames[it->first] ;
    253          names.push_back(name.c_str()) ;
    254     }
    255 }
    256 #else
    257 void U4Material::GetPropertyNames( std::vector<std::string>& names, const G4Material* mat )
    258 {
    259   const G4MaterialPropertiesTable* mpt_ = mat->GetMaterialPropertiesTable();
    260   if( mpt_ == nullptr ) return ;
    261   std::vector<G4String> pnames = mpt_->GetMaterialPropertyNames();
    262   //const vector<G4String>::const_iterator iter = pnames.begin();
    263   for(std::vector<G4String>::const_iterator iter = pnames.begin(); iter != pnames.end(); iter++)
    264     {
    265       // std::string name=(iter).c_str();
    266       //names.push_back(name);  
    267     }
    268 }
    269 #endif








::

    [ 10%] Building CXX object CMakeFiles/U4.dir/U4Random.cc.o


    In file included from /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:62,
                     from /data3/wenzel/newopticks_head/opticks/u4/U4VolumeMaker.cc:21:
    /data3/wenzel/newopticks_head/opticks/u4/U4MaterialPropertiesTable.h: 
    In static member function 
    ‘static void U4MaterialPropertiesTable::GetProperties(
        std::vector<std::__cxx11::basic_string<char> >&, 
        std::vector<G4PhysicsFreeVector*>&, 
        const G4MaterialPropertiesTable*)’:
    /data3/wenzel/newopticks_head/opticks/u4/U4MaterialPropertiesTable.h:165:27: 
        error: ‘const class G4MaterialPropertiesTable’ has no member named ‘GetPropertyMap’; did you mean ‘GetProperty’?
      165 |     const MIV* miv = mpt->GetPropertyMap();
          |                           ^~~~~~~~~~~~~~
          |                           GetProperty


More of same::

    [ 11%] Building CXX object CMakeFiles/U4.dir/U4Debug.cc.o
    [ 12%] Building CXX object CMakeFiles/U4.dir/U4Scintillation_Debug.cc.o
    In file included from /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:62,
                     from /data3/wenzel/newopticks_head/opticks/u4/U4Recorder.cc:28:
    /data3/wenzel/newopticks_head/opticks/u4/U4MaterialPropertiesTable.h: In static member function ‘static void U4MaterialPropertiesTable::GetProperties(std::vector<std::__cxx11::basic_string<char> >&, std::vector<G4PhysicsFreeVector*>&, const G4MaterialPropertiesTable*)’:
    /data3/wenzel/newopticks_head/opticks/u4/U4MaterialPropertiesTable.h:165:27: error: ‘const class G4MaterialPropertiesTable’ has no member named ‘GetPropertyMap’; did you mean ‘GetProperty’?
      165 |     const MIV* miv = mpt->GetPropertyMap();
          |                           ^~~~~~~~~~~~~~
          |                           GetProperty








U4Surface.h missing SNameOrder (probably version macro mixup)
------------------------------------------------------------------

Try fix, by adding the below to U4Surface.h::

     22 #include "S4.h"
     23 
     24 #if G4VERSION_NUMBER >= 1070
     25 #include "SNameOrder.h"
     26 #endif
     27 
     28 




::

    In file included from /data3/wenzel/newopticks_head/opticks/u4/U4VolumeMaker.cc:21:
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h: In static member function ‘static const std::vector<G4LogicalBorderSurface*>* U4Surface::PrepareBorderSurfaceVector(const G4LogicalBorderSurfaceTable*)’:
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:207:9: error: ‘SNameOrder’ was not declared in this scope
      207 |         SNameOrder<G4LogicalBorderSurface>::Sort( *vec, reverse, tail );
          |         ^~~~~~~~~~
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:207:42: error: expected primary-expression before ‘>’ token
      207 |         SNameOrder<G4LogicalBorderSurface>::Sort( *vec, reverse, tail );
          |                                          ^
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:207:45: error: ‘::Sort’ has not been declared
      207 |         SNameOrder<G4LogicalBorderSurface>::Sort( *vec, reverse, tail );
          |                                             ^~~~
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:208:55: error: expected primary-expression before ‘>’ token
      208 |         std::cout << SNameOrder<G4LogicalBorderSurface>::Desc( *vec ) << std::endl ;
          |                                                       ^
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:208:58: error: ‘::Desc’ has not been declared; did you mean ‘desc’?
      208 |         std::cout << SNameOrder<G4LogicalBorderSurface>::Desc( *vec ) << std::endl ;
          |                                                          ^~~~
          |                                                          desc



And again::

    In file included from /data3/wenzel/newopticks_head/opticks/u4/U4Recorder.cc:28:
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h: In static member function ‘static const std::vector<G4LogicalBorderSurface*>* U4Surface::PrepareBorderSurfaceVector(const G4LogicalBorderSurfaceTable*)’:
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:207:9: error: ‘SNameOrder’ was not declared in this scope
      207 |         SNameOrder<G4LogicalBorderSurface>::Sort( *vec, reverse, tail );
          |         ^~~~~~~~~~
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:207:42: error: expected primary-expression before ‘>’ token
      207 |         SNameOrder<G4LogicalBorderSurface>::Sort( *vec, reverse, tail );
          |                                          ^
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:207:45: error: ‘::Sort’ has not been declared
      207 |         SNameOrder<G4LogicalBorderSurface>::Sort( *vec, reverse, tail );
          |                                             ^~~~
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:208:55: error: expected primary-expression before ‘>’ token
      208 |         std::cout << SNameOrder<G4LogicalBorderSurface>::Desc( *vec ) << std::endl ;
          |                                                       ^
    /data3/wenzel/newopticks_head/opticks/u4/U4Surface.h:208:58: error: ‘::Desc’ has not been declared; did you mean ‘desc’?
      208 |         std::cout << SNameOrder<G4LogicalBorderSurface>::Desc( *vec ) << std::endl ;
          |                                                          ^~~~
          |                                                          desc
    make[2]: *** [CMakeFiles/U4.dir/build.make:146: CMakeFiles/U4.dir/U4VolumeMaker.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    make[2]: *** [CMakeFiles/U4.dir/build.make:160: CMakeFiles/U4.dir/U4Recorder.cc.o] Error 1
    make[1]: *** [CMakeFiles/Makefile2:920: CMakeFiles/U4.dir/all] Error 2
    make: *** [Makefile:146: all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /data3/wenzel/newopticks_head/local/opticks/build/u4 : non-zero rc 2
    === om-one-or-all install : non-zero rc 2
    === opticks-full : ERR from opticks-full-make




