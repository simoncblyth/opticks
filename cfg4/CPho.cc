#include <sstream>
#include "CPho.hh"

CPho::CPho(unsigned gs_, unsigned ix_, unsigned id_, bool re_)
    :
    gs(gs_),
    ix(ix_),
    id(id_),
    re(re_)
{
}

std::string CPho::desc() const 
{ 
    std::stringstream ss ; 
    ss << "CPho"
       << " gs " << gs
       << " ix " << ix
       << " id " << id
       << " re " << re
       ;  
    std::string s = ss.str(); 
    return s ; 
}


