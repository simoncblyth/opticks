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

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include <string>

// minimal node class standin for nnode used to develop tree machinery 

struct NPY_API no
{
    static no* copy( const no* a ); // retaining vtable of subclass instances 
    no* make_copy() const ;         // retaining vtable of subclass instances

    const char* label ; 
    no* left ; 
    no* right ; 
    no* parent ; 

    unsigned depth ;
    unsigned subdepth ;
    OpticksCSG_t type ; 
    bool    complement ; 
 
    std::string tag() const ;
    std::string id() const ;
    std::string desc() const ;
    unsigned maxdepth() const ;
    unsigned _maxdepth(unsigned depth) const;   // recursive 
    static unsigned nmaxu(const unsigned a, const unsigned b);

    bool is_primitive() const ; 
    bool is_bileaf() const ; 
    bool is_operator() const ; 
    bool is_zero() const ; 

    bool is_lrzero() const ;  //  l-zero AND  r-zero
    bool is_rzero() const ;   // !l-zero AND  r-zero
    bool is_lzero() const ;   //  l-zero AND !r-zero

    static no* make_node( OpticksCSG_t type, no* left=NULL, no* right=NULL );  
    static no* make_operator(OpticksCSG_t operator_, no* left=NULL, no* right=NULL );

}; 


inline no* no::make_operator(OpticksCSG_t operator_, no* left, no* right )
{
    no* node = make_node(operator_, left , right );
    return node ; 
}




