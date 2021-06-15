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

#include "OPTICKS_LOG.hh"
#include "SGDML.hh"


struct Demo
{
   int answer ;
};


void test_GenerateName()
{
    Demo* d = new Demo { 42 } ; 
    LOG(info) << SGDML::GenerateName( "Demo", d, true );
}



void test_Strip()
{
    LOG(info) ; 
    std::vector<std::string> names  = { "hell0xc0ffee" , "world0xdeadbeef", "hello0xworld0xcruel", "0xhello", "name_without_0X_lowercase",  } ; 
    std::vector<std::string> xnames = { "hell" ,         "world",           "hello",               ""       , "name_without_0X_lowercase"} ; 
    
    for(int i=0 ; i < int(names.size()) ; i++)
    {
       const std::string& xname = xnames[i] ; 
       const std::string& name = names[i] ; 
       std::string sname = SGDML::Strip(name) ; 
       std::string sname2 = SGDML::Strip(name.c_str()) ; 

       std::cout 
           << std::setw(3) << i 
           << " : "
           << std::setw(50) << name
           << " : "
           << std::setw(50) << sname
           << std::endl
           ;

        assert( strcmp(xname.c_str(), sname.c_str()) == 0 );  
        assert( strcmp(sname.c_str(), sname2.c_str()) == 0 );  
    }
}



int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv);

    test_GenerateName(); 
    test_Strip(); 
 
    return 0 ;
}   

// om-;TEST=SGDMLTest om-t 

