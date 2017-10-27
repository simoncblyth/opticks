#include <cstring>
#include <cassert>
#include <iostream>

#include "SAr.hh"

SAr::SAr( int argc_ , char** argv_ ) 
    :
    _argc( argc_ ),
    _argv( new char*[argc_] )
{
    assert( argc_ < 100 && "argc_ sanity check " );
    for(int i=0 ; i < argc_ ; i++ ) _argv[i] = strdup(argv_[i]) ; 
    //dump();
}


void SAr::dump()
{
    std::cout << "SAr _argc " << _argc << " ( " ; 
    for(int i=0 ; i < _argc ; i++ ) std::cout << " " << _argv[i] ; 
    std::cout << " ) " << std::endl ;  
} 



