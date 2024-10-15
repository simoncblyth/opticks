#pragma once
/**
SCF.h : Lightweight access to CSGFoundry geometry loaded from CFBASE directory
================================================================================

Using this from the A side is distinctly dodgy as it relies on the CFBASE 
directory geometry matching the current geometry.

Use from the B side is more acceptable as B side running is a debugging 
exercise subservient to A side running so in that situation can rely on 
CFBASE directory and it is then up to the user to ensure in the A-B matching 
that are using an appropriate CFBASE directory that matches. 

Could also argue that might as well use the CSG/CSGFoundry geometry rather than 
this lightweight alternative which can be loaded on the B side and it always
created on the A side anyhow ? 

**/


#include <string>
#include <sstream>
#include <vector>
#include "NP.hh"
#include "SSys.hh"
#include "SEventConfig.hh"
#include "SName.h"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"


struct SCF
{
    static SCF* INSTANCE ; 
    static SCF* Create(); 
    static SCF* Get() ; 



    static void ReadNames(const char* path_, std::vector<std::string>& names ); 

    template<typename T>
    static void LoadVec( std::vector<T>& vec, const char* dir, const char* name ); 

    static int  Index(const char* name, const std::vector<std::string>& names, unsigned max_count ); 

    std::vector<std::string> bnd ; 
    std::vector<std::string> msh ; 
    std::vector<std::string> pri ; 
    std::vector<qat4>       inst ;

    SCF(); 
    std::string desc() const ; 

    int getPrimIdx( const char* soname) const ; 
    int getMeshIdx( const char* soname) const ;  
    int getBoundary(const char* spec)   const ;  

    const qat4* getInst(unsigned instIdx) const ;
    const qat4* getInputPhotonFrame(const char* ipf_) const ; 
    const qat4* getInputPhotonFrame() const ; 

}; 


SCF* SCF::INSTANCE = nullptr ; 
SCF* SCF::Create()
{
    bool has_CFBASE = SSys::hasenvvar("CFBASE") ; 
    //if(!has_CFBASE) std::cerr << "SCF::Create BUT no CFBASE envvar " << std::endl ; 
    return has_CFBASE ? new SCF : nullptr ;  
}
SCF* SCF::Get(){  return INSTANCE ; }


SCF::SCF()
{
    ReadNames( "$CFBASE/CSGFoundry/SSim/bnd_names.txt", bnd ); 
    ReadNames( "$CFBASE/CSGFoundry/meshname.txt", msh ); 
    ReadNames( "$CFBASE/CSGFoundry/primname.txt", pri ); 
    LoadVec<qat4>( inst, "$CFBASE/CSGFoundry", "inst.npy" ); 
    INSTANCE = this ; 
}

std::string SCF::desc() const
{
    std::stringstream ss ; 
    ss << "SCF::desc"
       << " bnd.size " << bnd.size() 
       << " msh.size " << msh.size() 
       << " pri.size " << pri.size() 
       << " inst.size " << inst.size() 
       ;

    std::string s = ss.str();
    return s ; 
}

void SCF::ReadNames(const char* path_, std::vector<std::string>& names ) // static
{
    const char* path = SPath::Resolve(path_, NOOP); 
    NP::ReadNames(path, names); 
    std::cout << "SCF::ReadNames " << "path " << path << " names.size " << names.size() << std::endl ;
    for(unsigned i=0 ; i < names.size() ; i++) std::cout << std::setw(4) << i << " : " << names[i] << std::endl ;  
}

template<typename T>
void SCF::LoadVec( std::vector<T>& vec, const char* dir_, const char* name )
{
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    NP* a = NP::Load(dir, name);
    if( a == nullptr ) std::cout << "SCF::LoadVec FAIL dir_ " << dir_ << " name " << name << std::endl; 
    if( a == nullptr ) return ; 
    assert(a); 
    assert( a->shape.size()  == 3 ); 
    unsigned ni = a->shape[0] ; 
    //unsigned nj = a->shape[1] ; 
    //unsigned nk = a->shape[2] ; 

    vec.clear(); 
    vec.resize(ni); 
    memcpy( vec.data(),  a->bytes(), sizeof(T)*ni ); 
}





int SCF::Index(const char* name, const std::vector<std::string>& names, unsigned max_count )
{
    unsigned count = 0 ; 
    int index = NP::NameIndex( name, count, names );
    assert( max_count == 0 || count <= max_count );  
    return index ; 
}

/**
SCF::getPrimIdx
------------------------

HMM: this will not match Opticks in full geometry where meshnames 
appear repeatedly for many prim. 

HMM: potentially with live running could fix this by holding origin 
pointers to maintain the source G4VPhysicalVolume for every CSGPrim ?  

This would require the Geant4 SCFTest to do a translation to 
CSG on the fly and use that.  Given the heavy dependencies of 
the translation currently this solution not convenient.  

This is a capability that needs to wait for the new more direct G4->CSG "Geo" impl.
The as yet uncreated "Geo" full node tree needs to retain the connection
to the origin physical volumes and copyNo which needs to be carried into 
the CSG model : possibly with just the nodeindex. 
Then in SCF can reproduce the identity.    

**/

int SCF::getPrimIdx( const char* soname) const { return Index(soname, pri, 0 ); }
int SCF::getMeshIdx( const char* soname) const { return Index(soname, msh, 1 ); }
int SCF::getBoundary(const char* spec) const {   return Index(spec,   bnd, 1 ); }

const qat4* SCF::getInst(unsigned instIdx)   const { return instIdx  < inst.size()  ? inst.data()  + instIdx  : nullptr ; }

const qat4* SCF::getInputPhotonFrame(const char* ipf_) const 
{
    int ipf = ipf_ ? SName::ParseIntString(ipf_, -1) : -1 ; 
    const qat4* q = ipf > -1 ? getInst( ipf ) : nullptr ;
    return q ; 
}
const qat4* SCF::getInputPhotonFrame() const 
{
    const char* ipf_ = SEventConfig::InputPhotonFrame(); 
    return getInputPhotonFrame(ipf_); 
}



