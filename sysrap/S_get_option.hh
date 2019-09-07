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

#pragma once
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

/**
opticks/sysrap/S_get_option.hh
================================

Really minimal command line option parsing in this simple header::

    #include "S_get_option.hh"

    int main(int argc, char **argv)
    {
      const char* size_default = "1200,800" ; 

      int stack = get_option<int>(argc, argv, "--stack", "3000" );   
      int width = get_option<int>(argc, argv, "--size,0", size_default ) ;   
      int height = get_option<int>(argc, argv, "--size,1", size_default ) ;   

      std::cout << " stack " << stack << std::endl ; 
      std::cout << " width [" << width << "]" << std::endl ; 
      std::cout << " height [" << height << "]" << std::endl ; 

      return 0
    }

**/


const char* get_option_(int argc, char** argv, const char* option, const char* fallback)
{
    const char* val = NULL ; 
    for(int i=1 ; i < argc - 1 ; i++ ) 
    {   
        const char* a0 = argv[i] ; 
        const char* a1 = argv[i+1] ;
        if(a0 && strcmp(a0, option) == 0) val = a1 ;   
    }
    return val ? val : fallback ; 
}

template <typename T>
T lexical_cast(const char* par)
{
    T var;
    std::istringstream iss;
    iss.str(par);
    iss >> var;
    return var;
}

std::string get_field( const std::string& par, int field=-1, char delim=',')
{
    if( field < 0 ) return par ; 
    std::vector<std::string> fields ; 
    std::stringstream ss ; 
    int l = strlen(par.c_str()); 
    for(int i=0 ; i < l ; i++)
    {
         char c = par[i];  
         if( c != delim )
         {
             ss << c ; 
         }

         if( c == delim || i == l-1 )
         {
             std::string s = ss.str();  
             fields.push_back( s ); 
             ss.str("");
             ss.clear();
         }   
    }
    return field < int(fields.size()) ? fields[field] : "" ;  
}

template <typename T> T get_option( int argc, char** argv, const char* option, const char* fallback  )
{
    std::string opt = option ; 
    char delim = ',' ; 
    
    size_t pos = opt.find(delim) ;  
    int field(-1) ; 

    //std::cout << " pos " << pos << std::endl ; 
    if( pos != std::string::npos ) 
    {
        std::string s_field = opt.substr(pos+1) ; 
        opt = opt.substr(0, pos) ; 
        field = lexical_cast<int>( s_field.c_str() ); 

        //std::cout << " opt " << opt << " field " << field << std::endl ; 
    } 

    std::string par = get_option_( argc, argv, opt.c_str() , fallback ); 
    if( field > -1 ) par = get_field( par, field, delim ); 
    return lexical_cast<T>(par.c_str()) ; 
}


