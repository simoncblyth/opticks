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

#include <cassert>
#include <sstream>
#include "Ctrl.hh"


Ctrl::Ctrl(float* fptr, unsigned n) : num_cmds(0)
{
    assert( n == 4 ); 
    for(unsigned j=0 ; j < n ; j++ ) fc.f[j] = *(fptr+j) ; 


    for(int i=0 ; i < 8 ; i++)
    {
        char* p = fc.c + i*2 ;
        // retain the positional info, for possible timing control 
        if(*p == 0) 
        {
            cmds.push_back("  ");    
        } 
        else
        {
            std::string cmd(p, 2) ;
            cmds.push_back(cmd); 
            num_cmds += 1 ;  
        }
    }
}  


std::string Ctrl::getCommands() const 
{
    std::stringstream ss ; 
    unsigned num = cmds.size() ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        ss 
           << cmds[i] 
           << ( i < num - 1 ? "," : "")
        ; 
    }
    return ss.str();
}



