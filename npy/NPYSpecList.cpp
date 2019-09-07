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

#include <iterator>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <sstream>

#include "NPYSpecList.hpp"
#include "NPYSpec.hpp"


NPYSpecList::NPYSpecList()
{
}

void NPYSpecList::add( unsigned idx, const NPYSpec* spec )
{   
    m_idx.push_back(idx); 
    m_spec.push_back(spec); 
}

unsigned NPYSpecList::getNumSpec() const
{
    assert( m_idx.size() == m_spec.size()); 
    return m_idx.size(); 
}

const NPYSpec* NPYSpecList::getByIdx(unsigned idx) const 
{
    typedef std::vector<unsigned>::const_iterator IT ; 
    IT it = std::find( m_idx.begin() , m_idx.end(), idx );
    return it == m_idx.end() ? NULL : m_spec[ std::distance( m_idx.begin(), it ) ] ; 
}
        
std::string NPYSpecList::description() const 
{
    std::stringstream ss ; 
    unsigned num_spec = getNumSpec(); 
    for(unsigned i=0 ; i < num_spec ; i++)
    {
        unsigned idx = m_idx[i] ; 
        const NPYSpec* spec = m_spec[i] ; 
        ss 
           << std::setw(5) << idx 
           << " : "
           << spec->description()
           << std::endl 
          ; 
    }
    return ss.str(); 
}
 
