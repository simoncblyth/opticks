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
    
    // UID
    // 000  ___ 
    // 001  __D
    // 010  _I_ 
    // 011  _ID
    // 100  U__ 
    // 101  U_D
    // 110  UI_
    // 111  UID

    std::vector<unsigned> masks = {{ 
         0u,
         CSG::Mask(CSG_DIFFERENCE), 
         CSG::Mask(CSG_INTERSECTION), 
         CSG::Mask(CSG_INTERSECTION) | CSG::Mask(CSG_DIFFERENCE),
         CSG::Mask(CSG_UNION), 
         CSG::Mask(CSG_UNION) | CSG::Mask(CSG_DIFFERENCE),
         CSG::Mask(CSG_UNION) | CSG::Mask(CSG_INTERSECTION),
         CSG::Mask(CSG_UNION) | CSG::Mask(CSG_INTERSECTION) | CSG::Mask(CSG_DIFFERENCE)
    }}; 

    for(unsigned i=0 ; i < masks.size() ; i++)
    {
        unsigned mask = masks[i] ; 
        std::cout 
            << " i " << std::setw(5) << i
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
    

void test_OffsetType()
{
    std::vector<unsigned> types = { 
            CSG_ZERO, 
            CSG_TREE, 
                CSG_UNION, 
                CSG_INTERSECTION, 
                CSG_DIFFERENCE, 
            CSG_LIST, 
                CSG_CONTIGUOUS, 
                CSG_DISCONTIGUOUS,
                CSG_OVERLAP,
            CSG_LEAF,
                CSG_SPHERE,
                CSG_BOX,
                CSG_ZSPHERE,
                CSG_TUBS,
                CSG_CYLINDER,
                CSG_SLAB,
                CSG_PLANE,
                CSG_CONE,
                CSG_MULTICONE,
                CSG_BOX3,
                CSG_TRAPEZOID,
                CSG_CONVEXPOLYHEDRON,
                CSG_DISC,
                CSG_SEGMENT,
                CSG_ELLIPSOID,
                CSG_TORUS,
                CSG_HYPERBOLOID,
                CSG_CUBIC,
                CSG_INFCYLINDER,
                CSG_PHICUT, 
                CSG_THETACUT, 
                CSG_UNDEFINED
       }; 

    for(unsigned i=0 ; i < types.size() ; i++)
    {
         OpticksCSG_t type = (OpticksCSG_t)types[i]; 
         const char* name = CSG::Name(type) ; 
         OpticksCSG_t type2 = CSG::TypeCode(name); 
         const char* name2 = CSG::Name(type2) ; 

         unsigned offset_type = CSG::OffsetType(type); 
         unsigned type3 = CSG::TypeFromOffsetType( offset_type ); 

         std::cout 
              << " i " << std::setw(3) << i 
              << " type " << std::setw(3) << type
              << " offset_type " << std::setw(3) << offset_type
              << " CSG::Tag(type) " << std::setw(10) << CSG::Tag(type)
              << " CSG::Name(type) " << std::setw(15) << name
              << " CSG::Name(type2) " << std::setw(15) << name2
              << " CSG::IsPrimitive(type) " << std::setw(2) << CSG::IsPrimitive(type)
              << " CSG::IsList(type) " << std::setw(2) << CSG::IsList(type)
              << " CSG::IsCompound(type) " << std::setw(2) << CSG::IsCompound(type)
              << " CSG::IsLeaf(type) " << std::setw(2) << CSG::IsLeaf(type)
              << std::endl
              ;
         assert( type2 == type ); 
         assert( type3 == type ); 
    }
}

void test_MaskString()
{
    unsigned mask = CSG::Mask(CSG_SPHERE) | CSG::Mask(CSG_UNION)  ; 
    std::cout << CSG::MaskString(mask) << std::endl ; 

    unsigned typemask = 0 ; 

    for(unsigned i=0 ; i < 32 ; i++) 
    {
        OpticksCSG_t type = (OpticksCSG_t)CSG::TypeFromOffsetType(i) ; 
        typemask |= CSG::Mask(type); 

        const char* name =  CSG::Name(type)  ; 
        if( name == nullptr ) continue ; 
      
        std::cout 
            << " i " << std::setw(3) << i 
            << " type " << std::setw(3) << type
            << " CSG::Name(type) " << std::setw(20) << CSG::Name(type)
            << " typemask " << std::setw(10) << typemask 
            << " CSG::MaskString(typemask) " << CSG::MaskString(typemask)
            << std::endl 
            ;

    }


}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    /*
    test_type(); 
    test_HintCode(); 
    test_TypeMask(); 
    test_OffsetType(); 
    */
  
    test_MaskString(); 

    return 0 ; 
} 

