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

#include <cstddef>
#include "NOctNodeEnum.hpp"

const char* NOctNodeEnum::NOCT_ZERO_     = "ZERO" ;
const char* NOctNodeEnum::NOCT_INTERNAL_ = "INTERNAL" ;
const char* NOctNodeEnum::NOCT_LEAF_     = "LEAF" ;

const char* NOctNodeEnum::NOCTName(NOctNode_t type)
{
    switch(type)
    {
       case NOCT_ZERO:     return NOCT_ZERO_     ; break ; 
       case NOCT_INTERNAL: return NOCT_INTERNAL_ ; break ; 
       case NOCT_LEAF:     return NOCT_LEAF_     ; break ; 
    }
    return NULL ; 
}




