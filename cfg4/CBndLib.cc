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

#include "OpticksHub.hh"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
//#include "GSurLib.hh"

#include "CBndLib.hh"

CBndLib::CBndLib(OpticksHub* hub)
    :
    m_hub(hub),
    m_blib(m_hub->getBndLib()),
    m_mlib(m_hub->getMaterialLib()),
    m_slib(m_hub->getSurfaceLib())
    //m_surlib(m_hub->getSurLib())  // invokes the deferred GGeo::createSurLib, if CDetector didnt invoke 1st 
{
}


unsigned CBndLib::addBoundary(const char* spec)
{
    return m_blib->addBoundary(spec); 
}

GMaterial* CBndLib::getOuterMaterial(unsigned boundary)
{
    unsigned omat_ = m_blib->getOuterMaterial(boundary);
    GMaterial* omat = m_mlib->getMaterial(omat_);
    return omat ; 
}

GMaterial* CBndLib::getInnerMaterial(unsigned boundary)
{
    unsigned imat_ = m_blib->getInnerMaterial(boundary);
    GMaterial* imat = m_mlib->getMaterial(imat_);
    return imat ; 
}


GPropertyMap<double>* CBndLib::getOuterSurface(unsigned boundary)
{
    unsigned osur_ = m_blib->getOuterSurface(boundary);
    return m_slib->getSurface(osur_);
}
GPropertyMap<double>* CBndLib::getInnerSurface(unsigned boundary)
{
    unsigned isur_ = m_blib->getInnerSurface(boundary);
    return m_slib->getSurface(isur_);
}





/*
GSur* CBndLib::getOuterSurface(unsigned boundary)
{
    unsigned osur_ = m_bndlib->getOuterSurface(boundary);
    return m_surlib->getSur(osur_);
}
GSur* CBndLib::getInnerSurface(unsigned boundary)
{
    unsigned isur_ = m_bndlib->getInnerSurface(boundary);
    return m_surlib->getSur(isur_);
}
*/



