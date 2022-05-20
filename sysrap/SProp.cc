
#include <vector>
#include "PLOG.hh"

#include "NP.hh"
#include "SPath.hh"
#include "SProp.hh"


const plog::Severity SProp::LEVEL = PLOG::EnvLevel("SProp", "DEBUG"); 

const char* SProp::DEMO_PATH = "$IDPath/GScintillatorLib/LS_ori/RINDEX.npy" ;

/**
SProp::MockupCombination
------------------------

Mockup a real set of multiple properties, by loading 
a single property, copying it, and applying scalings. 

The source property is assumed to be provided in double precision 
(ie direct from Geant4 originals) with energies in MeV which are scaled to eV.
Also the properties are narrowed to float when the template type is float.

**/



const NP* SProp::MockupCombination(const char* path_ )  // static 
{
    const char* path = SPath::Resolve(path_ , NOOP); 
    LOG(LEVEL) 
        << "path_ " << path_  
        << "path " << path  
        ;

    if( path == nullptr ) return nullptr ;  // malformed path ?
    NP* a = NP::Load(path) ; 
    if( a == nullptr ) return nullptr ;  // non-existing path 

    bool is_double = strcmp( a->dtype, "<f8") == 0; 
    if(is_double == false) LOG(fatal) << "EXPECTING double precision input array " ; 
    assert(is_double); 

    return MockupCombination(a) ; 
}


const NP* SProp::MockupCombination(const NP* a_ )
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

    LOG(LEVEL) 
        << " com " << ( com ? com->desc() : "-" )
        ;

    return com ; 
}





/**
SProp::NarrowCombine
-------------------

Only implemented for float template specialization.

Combination using NP::Combine which pads shorter properties
allowing all to be combined into a single array, with final 
extra column used to record the payload column count.


HMM: maybe simpler to just MakeNarrow on the combined array ?

**/


const NP* SProp::NarrowCombine(const std::vector<const NP*>& aa )   // static
{
    LOG(LEVEL) << " narrowing double to float " ; 
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



const NP* SProp::Combine(const std::vector<const NP*>& aa )   // static
{
    LOG(LEVEL) << " not-narrowing retaining double " ; 
    NP* com = NP::Combine(aa) ;
    return com ;  
}





