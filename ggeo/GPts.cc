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


#include <iostream>
#include <csignal>

#include "SCount.hh"
#include "PLOG.hh"
#include "NPY.hpp"
#include "BStr.hh"
#include "GItemList.hh"

#include "GPt.hh"
#include "GPts.hh"


#include "PLOG.hh"

const plog::Severity GPts::LEVEL = PLOG::EnvLevel("GPts", "DEBUG") ; 


const char* GPts::GPTS_LIST = "GPts" ; 

GPts* GPts::Make()  // static
{
    LOG(LEVEL) ; 
    NPY<int>* ipt = NPY<int>::make(0, 4 ) ; 
    NPY<float>* plc = NPY<float>::make(0, 4, 4 ) ; 
    GItemList* specs = new GItemList(GPTS_LIST, "") ; 
    return new GPts(ipt, plc, specs); 
}

GPts* GPts::Load(const char* dir)  // static
{
    LOG(LEVEL) << dir ; 
    NPY<int>* ipt = LoadBuffer<int>(dir, "ipt") ; 
    NPY<float>* plc = LoadBuffer<float>(dir, "plc"); 
    GItemList* specs = GItemList::Load(dir, GPTS_LIST, "") ; 
    GPts* pts = new GPts(ipt, plc, specs); 
    pts->import(); 
    return pts ; 
}

/**
GPts::save
-----------

Exports from vector of GPt into transport array
and then writes the arrays to file.

**/

void GPts::save(const char* dir)
{
    LOG(LEVEL) << dir ; 
    export_();   
    if(m_ipt_buffer) m_ipt_buffer->save(dir, BufferName("ipt"));    
    if(m_plc_buffer) m_plc_buffer->save(dir, BufferName("plc"));    
    if(m_specs) m_specs->save(dir); 
}


template<typename T>
NPY<T>* GPts::LoadBuffer(const char* dir, const char* tag) // static
{
    const char* name = BufferName(tag) ;
    bool quietly = true ; 
    NPY<T>* buf = NPY<T>::load(dir, name, quietly ) ;
    return buf ; 
}

const char* GPts::BufferName(const char* tag) // static
{
    return BStr::concat(tag, "Buffer.npy", NULL) ;
}

GPts::GPts(NPY<int>* ipt, NPY<float>* plc, GItemList* specs) 
    :
    m_ipt_buffer(ipt),
    m_plc_buffer(plc),
    m_specs(specs)
{
}


/**
GPts::export_
--------------

From the vector of GPt instances into the transport arrays.

**/

void GPts::export_() // to the buffer
{
    for(unsigned i=0 ; i < getNumPt() ; i++ )
    {
        const GPt* pt = getPt(i); 
        glm::ivec4 ipt(pt->lvIdx, pt->ndIdx, pt->csgIdx, i); 

        m_specs->add(pt->spec.c_str());
        m_ipt_buffer->add(ipt); 
        m_plc_buffer->add(pt->placement) ;  
    }
}

/**
GPts::import
--------------

From transport arrays into the vector of newly instanciated 
GPt instances.

**/
 
void GPts::import()  
{
    assert( getNumPt() == 0 );  

    unsigned num_pt = m_specs->getNumItems(); 
    assert( num_pt == m_ipt_buffer->getShape(0)) ; 
    assert( num_pt == m_plc_buffer->getShape(0)) ; 

    for(unsigned i=0 ; i < num_pt ; i++)
    {
        const char* spec = m_specs->getKey(i); 
        glm::mat4 placement = m_plc_buffer->getMat4(i); 
        glm::ivec4 ipt = m_ipt_buffer->getQuadI(i); 

        int lvIdx = ipt.x ; 
        int ndIdx = ipt.y ; 
        int csgIdx = ipt.z ; 
  
        GPt* pt = new GPt( lvIdx, ndIdx, csgIdx, spec, placement ); 
        add(pt);  
    }
    assert( getNumPt() == num_pt );  
    LOG(LEVEL) << " num_pt " << num_pt ; 
}


unsigned GPts::getNumPt() const { return m_pts.size() ; } 

const GPt* GPts::getPt(unsigned i) const 
{
    assert( i < m_pts.size() ); 
    return m_pts[i] ; 
}

void GPts::add( GPt* pt )
{
    m_pts.push_back(pt);    
}



std::string GPts::brief() const 
{
    SCount count ;  // count occurrences of lvIdx 

    std::stringstream ss ; 
    unsigned num_pt = getNumPt() ; 
    unsigned edge = 10 ; 
    ss << " GPts.NumPt " << std::setw(5) << num_pt
       << " lvIdx (" 
        ;

    for( unsigned i=0 ; i < num_pt ; i++) 
    {
        const GPt* pt = getPt(i); 
        int lvIdx = pt->lvIdx ; 
        count.add(lvIdx); 

        if( num_pt > edge*2 )
        {
            if( i < edge || i > num_pt - edge ) ss << " " << lvIdx  ; 
            else if( i == edge )                ss << " " << "..."  ; 
        }
        else
        {
            ss << " " << lvIdx  ; 
        }
    } 
    ss << ")" ;
    ss << count.desc() ;     
    return ss.str(); 
}


void GPts::dump(const char* msg) const 
{
    LOG(info) << msg << brief() ; 
    for(unsigned i=0 ; i < getNumPt() ; i++ )
    {
        const GPt* pt = getPt(i); 
        std::cout 
            << " i " << std::setw(4) << i 
            << pt->desc()
            << std::endl 
            ; 
    }
}


template GGEO_API NPY<float>* GPts::LoadBuffer<float>(const char*, const char*) ;
template GGEO_API NPY<int>* GPts::LoadBuffer<int>(const char*, const char*) ;
template GGEO_API NPY<unsigned>* GPts::LoadBuffer<unsigned>(const char*, const char*) ;


