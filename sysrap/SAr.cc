/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cstring>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "SSys.hh"
#include "SAr.hh"

SAr* SAr::Instance = NULL ; 

SAr::SAr( const char* name , const char* envvar, char delim ) 
    :
    _argc(1), 
    _argv( new char*[1] ),
    _cmdline(NULL)
{
    _argv[0] = strdup(name) ; 

    init(envvar, delim); 
}

SAr::SAr( int argc_ , char** argv_ , const char* envvar, char delim ) 
    :
    _argc( argc_ ),
    _argv( argc_ > 0 ? new char*[argc_] : NULL ), 
    _cmdline(NULL)
{
    assert( _argc < 100 && "argc_ sanity check " );
    for(int i=0 ; i < argc_ ; i++ ) _argv[i] = strdup(argv_[i]) ; 

    init(envvar, delim); 
}

void SAr::init(const char* envvar, char delim)
{
    if(_argc == 0 )  // 0 means in-code not giving args
    {
        std::cout << "SAr::init _argc == 0  presumably from OPTICKS_LOG__(0,0) : args_from_envvar _argc " << _argc  << std::endl ; 
        args_from_envvar( envvar, delim) ; 
    }

    sanitycheck();

    std::string aline = argline();
    _cmdline = strdup(aline.c_str());    

    if(has_arg("--args"))
    {
        std::cout << _cmdline << std::endl ;     
    }

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
    for(int i=0 ; i < _argc ; i++ ) ss << ( _argv[i] ? _argv[i] : "NULL" ) << " "  ; 
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

int SAr::get_int_after(const char* option, const char* fallback) const 
{
    const char* arg = get_arg_after(option, fallback); 
    return SSys::atoi_(arg); 
}


bool SAr::has_arg( const char* arg ) const 
{
    for(int i=1 ; i < _argc ; i++ ) 
    {
        if(_argv[i] && strcmp(_argv[i], arg) == 0) return true ;   
    }    
    return false ; 
}
