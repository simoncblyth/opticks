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


#include <string>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "PLOG.hh"
#include "STranche.hh"


STranche::STranche(unsigned total_, unsigned max_tranche_) 
    :
    total(total_),
    max_tranche(max_tranche_),
    num_tranche((total+max_tranche-1)/max_tranche),
    last_tranche(total - (num_tranche-1)*max_tranche)  // <-- is max_tranche when  total % max_tranche == 0 
{
}

unsigned STranche::tranche_size(unsigned i) const 
{
    assert( i < num_tranche && " trance indices must be from 0 to tr.num_tranche - 1 inclusive  " ); 
    return i < num_tranche - 1 ? max_tranche : last_tranche  ; 
}
unsigned STranche::global_index(unsigned i, unsigned j ) const 
{
    return max_tranche*i + j ; 
}


const char* STranche::desc() const 
{
    std::stringstream ss ; 

    ss << "STranche"
       << " total " << total 
       << " max_tranche " << max_tranche 
       << " num_tranche " << num_tranche 
       << " last_tranche " << last_tranche 
       ;

    std::string s = ss.str();
    return strdup(s.c_str());
}


void STranche::dump(const char* msg)
{
    LOG(info) << msg << " desc " << desc() ; 

    unsigned cumsum = 0 ; 
    for(unsigned i=0 ; i < num_tranche ; i++)
    {
         unsigned size = tranche_size(i) ; 
         cumsum += size ; 

         unsigned global_index_0 = global_index(i, 0);
         unsigned global_index_1 = global_index(i, size-1); 

         std::cout << " i " << std::setw(6) << i 
                   << " tranche_size " << std::setw(6) << size
                   << " global_index_0 " << std::setw(6) << global_index_0
                   << " global_index_1 " << std::setw(6) << global_index_1
                   << " cumsum " << std::setw(6) << cumsum
                   << std::endl 
                   ;

         assert( cumsum == global_index_1 + 1 );
    }
    assert( cumsum == total ); 
}



