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

// TEST=NSolidTest om-t

#include <vector>
#include "OPTICKS_LOG.hh"
#include "NSolid.hpp"
#include "NNode.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    typedef std::vector<int> VI ; 
    VI lvs = { 18, 19, 20, 21 } ;  

    for( VI::const_iterator it=lvs.begin() ; it != lvs.end() ; it++ )
    { 
        int lv = *it ; 
        nnode* a = NSolid::create(lv); 
        assert(a && a->label);
       
        LOG(fatal) << "LV=" << lv << " label " << a->label ; 
        LOG(error) << a->ana_desc() ; 
    }


    return 0 ; 
} 
