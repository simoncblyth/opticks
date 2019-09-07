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
#include <string>
#include <cstring>
#include <cassert>

#include "BStr.hh"
#include "BBnd.hh"

#include "PLOG.hh"

const char BBnd::DELIM = '/' ; 


/**
BBnd::DuplicateOuterMaterial
------------------------------

Fabricate a boundary spec composed of just the outer material
from the argument spec.

This is used by NCSGList::createUniverse

**/

const char* BBnd::DuplicateOuterMaterial( const char* boundary0 )  // static 
{
    BBnd b(boundary0);
    return BBnd::Form(b.omat, NULL, NULL, b.omat);
}


/**
BBnd::Form
-----------

Form a spec string from arguments 

**/

const char* BBnd::Form(const char* omat_, const char* osur_, const char* isur_, const char* imat_)  // static 
{
    std::vector<std::string> uelem ;  
    uelem.push_back( omat_ ? omat_ : "" );
    uelem.push_back( osur_ ? osur_ : "" );
    uelem.push_back( isur_ ? isur_ : "" );
    uelem.push_back( imat_ ? imat_ : "" );

    std::string ubnd = BStr::join(uelem, DELIM ); 
    return strdup(ubnd.c_str());
}

/**
BBnd::BBnd
-----------

Populate the omat/osur/isur/imat struct by splitting the spec string 

**/

BBnd::BBnd(const char* spec)
{
    BStr::split( elem, spec, DELIM );
    bool four = elem.size() == 4  ;

    if(!four)
    LOG(fatal) << "BBnd::BBnd malformed boundary spec " << spec << " elem.size " << elem.size() ;  
    assert(four);

    omat = elem[0].empty() ? NULL : elem[0].c_str() ;
    osur = elem[1].empty() ? NULL : elem[1].c_str() ;
    isur = elem[2].empty() ? NULL : elem[2].c_str() ;
    imat = elem[3].empty() ? NULL : elem[3].c_str() ;

    assert( omat );
    assert( imat );  
}

std::string BBnd::desc() const 
{
    return BBnd::Form(omat, osur, isur, imat); 
}


