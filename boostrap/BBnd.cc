#include <vector>
#include <string>
#include <cstring>
#include <cassert>

#include "BStr.hh"
#include "BBnd.hh"

#include "PLOG.hh"

const char BBnd::DELIM = '/' ; 


const char* BBnd::DuplicateOuterMaterial( const char* boundary0 )
{
    BBnd b(boundary0);
    return BBnd::Form(b.omat, NULL, NULL, b.omat);
}


const char* BBnd::Form(const char* omat_, const char* osur_, const char* isur_, const char* imat_)
{
    std::vector<std::string> uelem ;  
    uelem.push_back( omat_ ? omat_ : "" );
    uelem.push_back( osur_ ? osur_ : "" );
    uelem.push_back( isur_ ? isur_ : "" );
    uelem.push_back( imat_ ? imat_ : "" );

    std::string ubnd = BStr::join(uelem, DELIM ); 
    return strdup(ubnd.c_str());
}


BBnd::BBnd(const char* spec)
{
    BStr::split( elem, spec, DELIM );
    bool four = elem.size() == 4  ;

    if(!four)
    LOG(fatal) << "BBnd::BBnd malformed boundary spec " << spec << " elem.size " << elem.size() ;  
    assert(four);

    omat = elem[0].empty() ? NULL : elem[0].c_str() ;
    osur = elem[1].empty() ? NULL : elem[1].c_str() ;
    isur = elem[2].empty() ? NULL : elem[2].c_str() ;
    imat = elem[3].empty() ? NULL : elem[3].c_str() ;

    assert( omat );
    assert( imat );  
}

std::string BBnd::desc() const 
{
    return BBnd::Form(omat, osur, isur, imat); 
}


