#include "s_csg.h"

s_csg* s_csg::INSTANCE = nullptr ; 
s_csg* s_csg::Get() { return INSTANCE ; } 


NPFold* s_csg::Serialize()
{
    assert(INSTANCE); 
    return INSTANCE ? INSTANCE->serialize() : nullptr ;   
}

void s_csg::Import(const NPFold* fold)
{
    if(INSTANCE == nullptr) new s_csg ; 
    assert( INSTANCE ) ; 

    int tot = INSTANCE->total_size() ; 
    if(tot != 0) std::cerr << INSTANCE->brief() ; 

    assert( tot == 0 ); 
    INSTANCE->import(fold); 
}





