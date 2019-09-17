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
#include "DummyGenstepsNPY.hpp"


NPY<float>* DummyGenstepsNPY::Make(unsigned num_gensteps)  // static
{
    DummyGenstepsNPY* gs = new DummyGenstepsNPY(num_gensteps);
    return gs->getNPY();
}

DummyGenstepsNPY::DummyGenstepsNPY(unsigned num_gensteps)
    :
    m_data(NPY<float>::make(num_gensteps, 6, 4))
{
    m_data->zero();   
    init();
}

void DummyGenstepsNPY::init()
{
    unsigned numGenstep = m_data->getNumItems();
    for(unsigned i=0 ; i < numGenstep ; i++)
    {   
         int fake_num_photons = 100 + i ; 
         nivec4 i0 = make_nivec4(i,i,i, fake_num_photons ) ;
         nvec4  f1 = make_nvec4(1000+i,1000+i,1000+i,1000+i) ;
         nvec4  f2 = make_nvec4(2000+i,2000+i,2000+i,2000+i) ;
         nvec4  f3 = make_nvec4(3000+i,3000+i,3000+i,3000+i) ;
         nvec4  f4 = make_nvec4(4000+i,4000+i,4000+i,4000+i) ;
         nvec4  f5 = make_nvec4(5000+i,5000+i,5000+i,5000+i) ;

         m_data->setQuadI( i0, i, 0 );
         m_data->setQuad( f1, i, 1 );
         m_data->setQuad( f2, i, 2 );
         m_data->setQuad( f3, i, 3 ) ; 
         m_data->setQuad( f4, i, 4 ) ; 
         m_data->setQuad( f5, i, 5 ) ; 
    }   
}

NPY<float>* DummyGenstepsNPY::getNPY()
{
    return m_data ; 
}


