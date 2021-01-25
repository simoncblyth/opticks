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
#include "DummyPhotonsNPY.hpp"


NPY<float>* DummyPhotonsNPY::Make(unsigned num_photons, unsigned hitmask, unsigned modulo, unsigned num_quad)  // static
{
    DummyPhotonsNPY* dp = new DummyPhotonsNPY(num_photons, hitmask, modulo, num_quad );
    return dp->getNPY();
}


DummyPhotonsNPY::DummyPhotonsNPY(unsigned num_photons, unsigned hitmask, unsigned modulo, unsigned num_quad)
    :
    m_data(NPY<float>::make(num_photons, num_quad, 4)),
    m_hitmask(hitmask),
    m_modulo(modulo),
    m_num_quad(num_quad)
{
    assert( m_num_quad == 6 || m_num_quad == 4 || m_num_quad == 2 ); 
    m_data->zero();   
    init();
}

void DummyPhotonsNPY::init()
{
    unsigned numHit(0);
    unsigned numPhoton = m_data->getNumItems();
    for(unsigned i=0 ; i < numPhoton ; i++)
    {   
        unsigned uhit = i % m_modulo == 0 ? m_hitmask  : 0  ;   // one in 10 are mock "hits"  
        if(uhit & m_hitmask ) numHit += 1 ; 

        if( m_num_quad == 4 )
        {
            nvec4 q0 = make_nvec4(i,i,i,i) ;
            nvec4 q1 = make_nvec4(1000+i,1000+i,1000+i,1000+i) ;
            nvec4 q2 = make_nvec4(2000+i,2000+i,2000+i,2000+i) ;
            nuvec4 u3 = make_nuvec4(3000+i,3000+i,3000+i,uhit) ;

            m_data->setQuad(  q0, i, 0 );
            m_data->setQuad(  q1, i, 1 );
            m_data->setQuad(  q2, i, 2 );
            m_data->setQuadU( u3, i, 3 );  // flags at the end

       } 
       else if( m_num_quad == 6 )
       {
            nvec4 q0 = make_nvec4(i,i,i,i) ;
            nvec4 q1 = make_nvec4(1000+i,1000+i,1000+i,1000+i) ;
            nvec4 q2 = make_nvec4(2000+i,2000+i,2000+i,2000+i) ;
            nvec4 q3 = make_nvec4(3000+i,3000+i,3000+i,3000+i) ;
            nvec4 q4 = make_nvec4(4000+i,4000+i,4000+i,4000+i) ;
            nuvec4 u5 = make_nuvec4(5000+i,5000+i,5000+i,uhit) ;

            m_data->setQuad(  q0, i, 0 );
            m_data->setQuad(  q1, i, 1 );
            m_data->setQuad(  q2, i, 2 );
            m_data->setQuad(  q3, i, 3 );
            m_data->setQuad(  q4, i, 4 );
            m_data->setQuadU( u5, i, 5 );  // flags at the end
       }
       else if( m_num_quad == 2 )
       {
            nvec4 q0 = make_nvec4(i,i,i,i) ;
            nuvec4 u1 = make_nuvec4(1000+i,1000+i,1000+i,uhit) ;

            m_data->setQuad(  q0, i, 0 );
            m_data->setQuadU( u1, i, 1 );  // flags at the end
       }
    }   
    m_data->setNumHit(numHit);
}

NPY<float>* DummyPhotonsNPY::getNPY()
{
    return m_data ; 
}


