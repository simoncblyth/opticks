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

#include <sstream>
#include <iomanip>
#include "NNode.hpp"
#include "NNodeCoincidence.hpp"

bool NNodeCoincidence::is_siblings() const
{
   return i->parent && j->parent && i->parent == j->parent ;
}

bool NNodeCoincidence::is_union_siblings() const
{
   return is_siblings() && i->parent->type == CSG_UNION ;
}

bool NNodeCoincidence::is_union_parents() const
{
   return i->parent && j->parent && i->parent->type == CSG_UNION && j->parent->type == CSG_UNION ;
}

std::string NNodeCoincidence::desc() const 
{
    std::stringstream ss ; 
    ss
        << "(" << std::setw(2) << i->idx
        << "," << std::setw(2) << j->idx
        << ")"
        << " " << NNodeEnum::PairType(p)
        << " " << NNodeEnum::NudgeType(n)
        << " " << i->tag()
        << " " << j->tag()
        << " sibs " << ( is_siblings() ? "Y" : "N" )
        << " u_sibs " << ( is_union_siblings() ? "Y" : "N" )
        << " u_par " << ( is_union_parents() ? "Y" : "N" )
        << " u_same " << ( nnode::is_same_union(i,j) ? "Y" : "N" )
        << " " << ( fixed ? "FIXED" : "" )
        ; 
  
    return ss.str();
}


