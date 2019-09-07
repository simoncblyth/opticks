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


#include "NPY.hpp"
#include "OpEvt.hh"


OpEvt::OpEvt() 
    :
    m_genstep(NULL)
{
}

void OpEvt::addGenstep( float* data, unsigned num_float )
{
    assert( num_float == 6*4 ) ;     
    if(!m_genstep) m_genstep = NPY<float>::make(0,6,4) ; 
    m_genstep->add(data, num_float ); 
}

unsigned OpEvt::getNumGensteps() const 
{
    return m_genstep ? m_genstep->getShape(0) : 0 ; 
}

NPY<float>* OpEvt::getEmbeddedGensteps()
{
    return m_genstep ; 
}



void OpEvt::loadEmbeddedGensteps(const char* path)
{
    m_genstep = NPY<float>::load(path) ; 
}

void OpEvt::saveEmbeddedGensteps(const char* path) const 
{
    if(!m_genstep) return ; 
    m_genstep->save(path) ; 
}


void OpEvt::resetGensteps() 
{
    m_genstep->reset();
    assert( getNumGensteps() == 0 );
}




