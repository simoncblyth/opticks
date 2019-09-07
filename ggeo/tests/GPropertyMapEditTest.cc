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

#include <vector>
#include <map>
#include <string>

#include "OPTICKS_LOG.hh"


struct Prop
{
    Prop(const char* name_, int value_)
        :
        name( strdup(name_) ),
        value(value_) 
    {
    }  
    const char* name ; 
    int value ; 
};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    std::map<std::string, Prop*> pm ; 

    Prop* a = new Prop("yo1", 42 ) ; 
    Prop* b = new Prop("yo2", 43 ) ; 
    Prop* c = new Prop("yo3", 44 ) ; 

    pm["ri"] = a ; 
    pm["vg"] = b ; 
    pm["sc"] = c ; 


    std::string a_key = "ri" ; 
    std::string b_key = "vg" ; 
    std::string c_key = "sc" ; 

    assert( pm.at(a_key) == a ) ;
    assert( pm.at(b_key) == b ) ;
    assert( pm.at(c_key) == c ) ;
 
    return 0 ; 
}
