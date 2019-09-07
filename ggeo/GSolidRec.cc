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

#include <iomanip>
#include <sstream>

#include "BStr.hh"
#include "NNode.hpp"
#include "NCSG.hpp"
#include "GSolidRec.hh"

GSolidRec::GSolidRec( const nnode* raw_, const nnode* balanced_, const NCSG* csg_, unsigned soIdx_, unsigned lvIdx_ )
    :
    raw(raw_),
    balanced(balanced_),
    csg(csg_),
    soIdx(soIdx_),
    lvIdx(lvIdx_)
{
}

GSolidRec::~GSolidRec()
{
}


std::string GSolidRec::desc() const 
{
    std::stringstream ss ; 
    ss
        << " so:" << BStr::utoa(soIdx, 3, true)  
        << " lv:" << BStr::utoa(lvIdx, 3, true)  
        << " rmx:" << BStr::utoa(raw->maxdepth(), 2, true )  
        << " bmx:" << BStr::utoa(balanced->maxdepth(), 2, true )  
        << " soName: " << csg->get_soname() 
        ;

    return ss.str(); 
}



