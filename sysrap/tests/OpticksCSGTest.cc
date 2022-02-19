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
#include <iostream>
#include <sstream>
#include <iomanip>

#include "OPTICKS_LOG.hh"
#include "OpticksCSG.h"


void test_type()
{
    LOG(info); 
    for(unsigned i=0 ; i < 100 ; i++)
    {
        OpticksCSG_t type = (OpticksCSG_t)i ; 
        if(!CSG::Exists(type)) continue ; 

        const char*  name = CSG::Name( type );

        std::cout 
                   << " type " << std::setw(3) << type
                   << " name " << std::setw(20) << name
                   << std::endl ; 


    }
}

void test_TypeMask()
{
    LOG(info); 

    std::vector<unsigned> masks = {{ 
         CSG::UnionMask(), 
         CSG::IntersectionMask(), 
         CSG::DifferenceMask(), 
         CSG::UnionMask() | CSG::IntersectionMask(),
         CSG::UnionMask() | CSG::IntersectionMask() | CSG::DifferenceMask()
    }}; 

    for(unsigned i=0 ; i < masks.size() ; i++)
    {
        unsigned mask = masks[i] ; 
        std::cout 
            << " mask " << std::setw(5) << mask 
            << " CSG::TypeMask(mask) " << std::setw(10) << CSG::TypeMask(mask)
            << " CSG::IsPositiveMask(mask) " << std::setw(2) << CSG::IsPositiveMask(mask)
            << std::endl 
            ; 
    }

}


void test_HintCode(const char* name)
{
     unsigned hintcode = CSG::HintCode(name); 
     std::cout 
         << " name " << std::setw(40) << name 
         << " hintcode " << std::setw(6) << hintcode
         << " CSG::Name(hintcode) " << std::setw(15) << CSG::Name(hintcode)
         << std::endl 
         ;

}

const char* NAMES = R"LITERAL(
Hello_CSG_CONTIGUOUS
Hello_CSG_DISCONTIGUOUS
Hello_CSG_OVERLAP
Name_without_any_hint
)LITERAL";

void test_HintCode()
{
    std::stringstream ss(NAMES) ;
    std::string name ;
    while (std::getline(ss, name)) if(!name.empty()) test_HintCode(name.c_str());     
}
    



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    /*
    test_type(); 
    test_TypeMask(); 
    */
    test_HintCode(); 
  

    return 0 ; 
} 

