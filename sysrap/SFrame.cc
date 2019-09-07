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

#include <iostream>
#include <iomanip>
#include <cxxabi.h>

#include "SFrame.hh"

SFrame::SFrame(char* line_)
    :
    line(line_),
    name(NULL),
    offset(NULL),
    end_offset(NULL),
    func(NULL)
{
    parse(); 
} 

SFrame::~SFrame()
{
    free(func);  
}

void SFrame::dump()
{
    std::ostream& out = std::cout ;
    dump(out);    
}



#ifdef __APPLE__
void SFrame::parse()
{
    /**
    4   libG4processes.dylib                0x00000001090baee5 _ZN12G4VEmProcess36PostStepGetPhysicalInteractionLengthERK7G4TrackdP16G4ForceCondition + 661 
    **/ 
    for( char *p = line ; *p ; ++p )
    {   
        if (( *p == '_' ) && ( *(p-1) == ' ' )) name = p-1;
        else if ( *p == '+' ) offset = p-1;
    }   

    if( name && offset && ( name < offset ))
    {   
        *name++ = '\0';    // plant terminator into line
        *offset++ = '\0';  // plant terminator into name  
        func = demangle(); 
    }  
}


void SFrame::dump(std::ostream& out)
{
     if( func )
     {
         out 
             << std::setw(30) << std::left << line  
             << " "
             << std::setw(100) << std::left << func
             << " "
             << std::setw(10) << std::left << offset 
             << " " 
             << std::endl 
             ;
     }
     else
     {
         out << line << std::endl ; 

     }
}

#else
void SFrame::parse()
{
    /**
    /home/blyth/local/opticks/externals/lib64/libG4tracking.so(_ZN10G4VProcess12PostStepGPILERK7G4TrackdP16G4ForceCondition+0x42) [0x7ffff36ff9b2] 
    **/
    for( char *p = line ; *p ; ++p )
    {   
        if ( *p == '(' ) name = p;
        else if ( *p == '+' ) offset = p;
        else if ( *p == ')' && ( offset || name )) end_offset = p;
    }   

    if( name && end_offset && ( name < end_offset ))
    {   
        *name++ = '\0';    // plant terminator into line
        *end_offset++ = '\0';  // plant terminator into name  
        if(offset) *offset++ = '\0' ; 
        func = demangle(); 
    }  
}
void SFrame::dump(std::ostream& out)
{
     if( func )
     {
         out 
               << std::setw(25) << std::left << ( end_offset ? end_offset : "" )  // addr
               << " " << std::setw(10) << std::left << offset 
               << " " << std::setw(60) << line  
               << " " << std::setw(60) << std::left << func
               << std::endl 
               ;
     }
     else
     {
         out  << line << std::endl ; 

     }
}
#endif

char* SFrame::demangle() // demangling fails for non C++ symbols
{
    int status;
    char* ret = abi::__cxa_demangle( name, NULL, NULL, &status );
    return status == 0 ? ret : NULL ; 
}



