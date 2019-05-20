#include <cstring>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "SAr.hh"

SAr* SAr::Instance = NULL ; 


SAr::SAr( int argc_ , char** argv_ , const char* envvar, char delim ) 
    :
    _argc( argc_ ),
    _argv( argc_ > 0 ? new char*[argc_] : NULL ), 
    _cmdline(NULL)
{
    if(argc_ == 0 )  // 0 means in-code not giving args
    {
        std::cout << "SAr::SAr argc_ == 0  presumably from OPTICKS_LOG__(0,0) : args_from_envvar argc_ " << argc_  << std::endl ; 
        args_from_envvar( envvar, delim) ; 
    }
    else
    { 
        assert( argc_ < 100 && "argc_ sanity check " );
        for(int i=0 ; i < argc_ ; i++ ) _argv[i] = strdup(argv_[i]) ; 
    }

    sanitycheck();

    std::string aline = argline();
    _cmdline = strdup(aline.c_str());    


    if(Instance)
        std::cout << "SAr::SAr replacing Instance " << std::endl ; 

    Instance = this ; 

    //dump();
}


void SAr::sanitycheck() const
{
    for(int i=0 ; i < _argc ; i++) 
    {
        const char* s = _argv[i] ; 
        if(strlen(s) > 3 && strncmp(s, "---", 3) == 0)
        {
            std::cout << "SAr::sanitycheck FAILURE for argument " << i << "[" << s << "]" << std::endl ; 
            assert(0 && "arguments starting with three dashes --- are not allowed ");    
        }  
    }

}


const char* SAr::exepath() const 
{
   return _argv ? _argv[0] : NULL  ;  
}
const char* SAr::exename() const
{
   return Basename(exepath()); 
}
const char* SAr::cmdline() const 
{
   return _cmdline ;  
}

const char* SAr::Basename(const char* path)
{
    if(!path) return NULL ; 
    const char *s = strrchr(path, '/') ;
    return s ? strdup(s+1) : strdup(path) ;  
}


void SAr::args_from_envvar( const char* envvar, char delim )
{
    const char* argline = envvar ? getenv(envvar) :  NULL ;

    if(argline == NULL) 
    {
        std::cout << "SAr::args_from_envvar but no argline provided " << std::endl ; 
        return ; 
    }

    std::cout << "SAr::args_from_envvar argline: " << argline << std::endl ; 

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



void SAr::dump() const 
{
    std::cout << "SAr::dump " ; 
    std::cout << "SAr _argc " << _argc << " ( " ; 
    for(int i=0 ; i < _argc ; i++ ) std::cout << " " << ( _argv[i] ? _argv[i] : "NULL" ) ; 
    std::cout << " ) " << std::endl ;  
} 


std::string SAr::argline() const 
{
    std::stringstream ss ; 
    for(int i=0 ; i < _argc ; i++ ) ss << " " << ( _argv[i] ? _argv[i] : "NULL" ) ; 
    return ss.str(); 
}

const char* SAr::get_arg_after(const char* option, const char* fallback) const
{
    for(int i=1 ; i < _argc - 1 ; i++ ) 
    {
        const char* a0 = _argv[i] ; 
        const char* a1 = _argv[i+1] ;
        if(a0 && strcmp(a0, option) == 0) return a1 ;   
    }
    return fallback ; 
}


