#include <cstring>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "SAr.hh"

SAr::SAr( int argc_ , char** argv_ , const char* envvar, char delim ) 
    :
    _argc( argc_ ),
    _argv( argc_ > 0 ? new char*[argc_] : NULL )
{
    if(argc_ == 0 || argc_ == 1 )  // 0 means in-code not giving args, 1 means just no arguments  
    {
        args_from_envvar( envvar, delim) ; 
    }
    else
    { 
        assert( argc_ < 100 && "argc_ sanity check " );
        for(int i=0 ; i < argc_ ; i++ ) _argv[i] = strdup(argv_[i]) ; 
    }
    //dump();
}


void SAr::args_from_envvar( const char* envvar, char delim )
{
    const char* argline = envvar ? getenv(envvar) :  NULL ;
    if(argline == NULL) return ; 

    std::stringstream ss; 
    ss.str(argline)  ;

    std::vector<std::string> args ; 
    args.push_back( envvar ) ;     // equivalent executable

    std::string s;
    while (std::getline(ss, s, delim)) args.push_back(s) ; 
    
    _argc = args.size(); 
    _argv = new char*[_argc] ; 

    for(int i=0 ; i < _argc ; i++ ) _argv[i] = strdup(args[i].c_str()) ; 
}



void SAr::dump()
{
    std::cout << "SAr _argc " << _argc << " ( " ; 
    for(int i=0 ; i < _argc ; i++ ) std::cout << " " << _argv[i] ; 
    std::cout << " ) " << std::endl ;  
} 



