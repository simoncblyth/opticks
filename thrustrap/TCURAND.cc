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

#include "TCURAND.hh"
#include "TCURANDImp.hh"
#include "PLOG.hh"

template <typename T>
const plog::Severity TCURAND<T>::LEVEL = PLOG::EnvLevel("TCURAND", "DEBUG") ; 

template <typename T>
TCURAND<T>::TCURAND(unsigned ni, unsigned nj, unsigned nk)
    :
    m_imp(new TCURANDImp<T>(ni, nj, nk))
{
}

template <typename T>
void TCURAND<T>::setIBase(unsigned ibase)
{
    LOG(LEVEL) << " ibase " << ibase ; 
    m_imp->setIBase(ibase); 
}

template <typename T>
unsigned TCURAND<T>::getIBase() const 
{
    return m_imp->getIBase(); 
}


template <typename T>
void TCURAND<T>::generate()
{
    m_imp->generate(); 
}

template <typename T>
NPY<T>* TCURAND<T>::getArray() const 
{
    return m_imp->getArray() ; 
}


template class TCURAND<float>;
template class TCURAND<double>;


