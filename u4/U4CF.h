#pragma once

#include <string>
#include <sstream>
#include <vector>
#include "NP.hh"
#include "SSys.hh"


struct U4CF
{
    static U4CF* INSTANCE ; 
    static U4CF* Create(); 
    static U4CF* Get() ; 
    static void ReadNames(const char* path_, std::vector<std::string>& names ); 
    static unsigned Index(const char* name, const std::vector<std::string>& names, unsigned max_count ); 

    std::vector<std::string> bnd ; 
    std::vector<std::string> msh ; 
    std::vector<std::string> pri ; 

    std::string desc() const ; 
    U4CF(); 

    unsigned getPrimIdx( const char* soname) const ; 
    unsigned getMeshIdx( const char* soname) const ;  
    unsigned getBoundary(const char* spec)   const ;  


}; 


U4CF* U4CF::INSTANCE = nullptr ; 
U4CF* U4CF::Create(){  return SSys::hasenvvar("CFBASE") ? new U4CF : nullptr ;  }
U4CF* U4CF::Get(){  return INSTANCE ; }

std::string U4CF::desc() const
{
    std::stringstream ss ; 
    ss << "U4CF::desc"
       << " bnd.size " << bnd.size()
       << " msh.size " << msh.size()
       << " pri.size " << pri.size()
       ;
    std::string s = ss.str(); 
    return s ; 
}


U4CF::U4CF()
{
    ReadNames( "$CFBASE/CSGFoundry/SSim/bnd_names.txt", bnd ); 
    ReadNames( "$CFBASE/CSGFoundry/meshname.txt", msh ); 
    ReadNames( "$CFBASE/CSGFoundry/primname.txt", pri ); 

    INSTANCE = this ; 
}
void U4CF::ReadNames(const char* path_, std::vector<std::string>& names ) // static
{
    const char* path = SPath::Resolve(path_, NOOP); 
    NP::ReadNames(path, names); 
    std::cout << "U4CF::ReadNames " << "path " << path << " names.size " << names.size() << std::endl ;
    for(unsigned i=0 ; i < names.size() ; i++) std::cout << std::setw(4) << i << " : " << names[i] << std::endl ;  
}

unsigned U4CF::Index(const char* name, const std::vector<std::string>& names, unsigned max_count )
{
    unsigned count = 0 ; 
    unsigned index = NP::NameIndex( name, count, names );
    assert( max_count == 0 || count <= max_count );  
    return index ; 
}

/**
U4CF::getPrimIdx
------------------------

HMM: this will not match Opticks in full geometry where meshnames 
appear repeatedly for many prim. 

HMM: potentially with live running could fix this by holding origin 
pointers to maintain the source G4VPhysicalVolume for every CSGPrim ?  

This would require the Geant4 U4CFTest to do a translation to 
CSG on the fly and use that.  Given the heavy dependencies of 
the translation currently this solution not convenient.  

This is a capability that needs to wait for the new more direct G4->CSG "Geo" impl.
The as yet uncreated "Geo" full node tree needs to retain the connection
to the origin physical volumes and copyNo which needs to be carried into 
the CSG model : possibly with just the nodeindex. 
Then in U4CF can reproduce the identity.    

**/

unsigned U4CF::getPrimIdx( const char* soname) const { return Index(soname, pri, 0 ); }
unsigned U4CF::getMeshIdx( const char* soname) const { return Index(soname, msh, 1 ); }
unsigned U4CF::getBoundary(const char* spec) const {   return Index(spec,   bnd, 1 ); }



