#pragma once
/**
SPropMockup.h : formerly SProp.hh
===================================

::

    epsilon:opticks blyth$ opticks-f SPropMockup.h 
    ./sysrap/CMakeLists.txt:    SPropMockup.h
    ./sysrap/SPropMockup.h:SPropMockup.h : formerly SProp.hh
    ./ggeo/GGeo.cc:#include "SPropMockup.h"  
    ./qudarap/tests/QPropTest.cc:#include "SPropMockup.h"
    epsilon:opticks blyth$ 

**/

#include <vector>
struct NP ; 

struct SPropMockup
{
    static constexpr const char* DEMO_BASE = "$HOME/.opticks/GEOM/$GEOM" ;
    static constexpr const char* DEMO_RELP = "GGeo/GScintillatorLib/LS_ori/RINDEX.npy" ; 

    static const NP* CombinationDemo(); 
    static const NP* Combination(const char* base, const char* relp ); 
    static const NP* Combination(const NP* a_ ); 

    // TODO: below functionality belongs in NP.hh not here 
    static const NP* NarrowCombine(const std::vector<const NP*>& aa ); 
    static const NP* Combine(const std::vector<const NP*>& aa ); 
}; 


#include <vector>
#include "NP.hh"
#include "spath.h"


inline const NP* SPropMockup::CombinationDemo() // static
{
    const NP* propcom = Combination( DEMO_BASE, DEMO_RELP);
    return propcom ;  
}

/**
SPropMockup::Combination
------------------------

Mockup a real set of multiple properties, by loading 
a single property, copying it, and applying scalings. 

The source property is assumed to be provided in double precision 
(ie direct from Geant4 originals) with energies in MeV which are scaled to eV.
Also the properties are narrowed to float when the template type is float.

**/


inline const NP* SPropMockup::Combination(const char* base, const char* relp )  // static 
{
    const char* path = spath::Resolve(base, relp); 
    std::cout
        << "SPropMockup::Combination"
        << " base " << base  
        << " relp " << relp  
        << " spath::Resolve to path " << path  
        << std::endl 
        ;

    if( path == nullptr ) return nullptr ;  // malformed path ?

    bool exists = NP::Exists(path) ; 
    std::cout
        << "SPropMockup::Combination"
        << " path " << ( path ? path : "-" )
        << " exists " << ( exists ? "YES" : "NO " )
        << std::endl 
        ;


    NP* a = exists ? NP::Load(path) : nullptr ; 
    if( a == nullptr ) return nullptr ;  // non-existing path 

    bool is_double = strcmp( a->dtype, "<f8") == 0; 
    if(!is_double) std::cerr 
       << "SPropMockup::Combination"
       << " EXPECTING double precision input array " 
       << std::endl 
       ; 
    assert(is_double); 

    return Combination(a) ; 
}


inline const NP* SPropMockup::Combination(const NP* a_ )
{
    NP* a = a_->copy(); 
    NP* b = a_->copy(); 
    NP* c = a_->copy(); 

    a->pscale<double>(1e6, 0u);   // energy scale from MeV to eV,   1.55 to 15.5 eV
    b->pscale<double>(1e6, 0u); 
    c->pscale<double>(1e6, 0u); 

    b->pscale<double>(1.05, 1u); 
    c->pscale<double>(0.95, 1u); 

    std::vector<const NP*> aa = {a, b, c } ; 
    const NP* com = NarrowCombine(aa); 

    std::cout
        << "SPropMockup::Combination"
        << " com " << ( com ? com->desc() : "-" )
        << std::endl 
        ;

    return com ; 
}


/**
SPropMockup::NarrowCombine
------------------------------

Only implemented for float template specialization.

Combination using NP::Combine which pads shorter properties
allowing all to be combined into a single array, with final 
extra column used to record the payload column count.


HMM: maybe simpler to just MakeNarrow on the combined array ?

**/


inline const NP* SPropMockup::NarrowCombine(const std::vector<const NP*>& aa )   // static
{
    std::cout << "SPropMockup::NarrowCombine : narrowing double to float " << std::endl ; 
    std::vector<const NP*> nn ; 
    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i] ; 
        const NP* n = NP::MakeNarrow( a );
        nn.push_back(n); 
    }
    NP* com = NP::Combine(nn) ; 
    return com ;  
}

inline const NP* SPropMockup::Combine(const std::vector<const NP*>& aa )   // static
{
    std::cout << "SPropMockup::Combine :  not-narrowing retaining double " << std::endl  ; 
    NP* com = NP::Combine(aa) ;
    return com ;  
}


