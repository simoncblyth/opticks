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

#include "THRAP_HEAD.hh"
#include "TUtil.hh"
#include "TRngBuf.hh"
#include "TCURANDImp.hh"
#include "THRAP_TAIL.hh"
#include "CDevice.hh"

#include "Opticks.hh"
#include "PLOG.hh"

template <typename T>
const plog::Severity TCURANDImp<T>::LEVEL = PLOG::EnvLevel("TCURANDImp", "DEBUG"); 
 

template <typename T>
int TCURANDImp<T>::preinit() 
{
    const char* dirpath = nullptr ; 
    bool nosave = true ; 

    CDevice::Visible(m_visible_device, dirpath, nosave );  
    CDevice::Dump(m_visible_device , "visible devices"); 

    OKI_PROFILE("_TCURANDImp::TCURANDImp"); 
    return 0 ; 
}
        
template <typename T>
TCURANDImp<T>::TCURANDImp( unsigned ni, unsigned nj, unsigned nk ) 
    :
    m_preinit(preinit()),
    m_ni(ni),
    m_nj(nj),
    m_nk(nk),
    m_elem( ni*nj*nk ),
    m_ox(NPY<T>::make( ni, nj, nk )),
    m_predox(predox()),
    m_dox(m_elem),
    m_postdox(postdox()),
    m_spec(make_bufspec<T>(m_dox)), 
    m_trb(new TRngBuf<T>( ni, nj*nk, m_spec ))
{
    init(); 
}


template <typename T>
void TCURANDImp<T>::init() 
{
    LOG(LEVEL) << desc() ;   
    m_ox->zero(); 
    OKI_PROFILE("TCURANDImp::TCURANDImp"); 
}


template <typename T>
int TCURANDImp<T>::predox() 
{
    LOG(LEVEL); 
    OKI_PROFILE("_dvec_dox"); 
    return 0 ; 
}

template <typename T>
int TCURANDImp<T>::postdox() 
{
    LOG(LEVEL); 
    OKI_PROFILE("dvec_dox"); 
    return 0 ; 
}



template <typename T>
std::string TCURANDImp<T>::desc()  const 
{
    std::stringstream ss ; 
    ss << "TCURANDImp"
       << " ox " << m_ox->getShapeString() 
       << " elem " << m_elem
       ; 
    return ss.str(); 
}


template <typename T>
void TCURANDImp<T>::setIBase(unsigned ibase)
{
    LOG(LEVEL) << " ibase " << ibase ;   
    m_trb->setIBase( ibase ); 
    generate(); 
}

template <typename T>
unsigned TCURANDImp<T>::getIBase() const 
{
    return m_trb->getIBase(); 
}





/**
TCURANDImp<T>::generate
-------------------------

GPU generation and download to host, updating m_ox array 

**/

template <typename T>
void TCURANDImp<T>::generate()
{
    m_trb->generate(); 
    bool verbose = true ; 
    m_trb->download( m_ox, verbose ) ; 
}

template <typename T>
NPY<T>* TCURANDImp<T>::getArray() const 
{
    return m_ox ; 
}


template class TCURANDImp<float>;
template class TCURANDImp<double>;


