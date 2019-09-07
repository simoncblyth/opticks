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


#include <cassert>

#include "Opticks.hh"
#include "OpticksResource.hh"

#include "NSlice.hpp"

#include "GMergedMesh.hh"
#include "GBndLib.hh"
#include "GPmtLib.hh"
#include "GPmt.hh"
#include "PLOG.hh"


const char* GPmtLib::TRI_PMT_PATH = "dpib/GMergedMesh/0" ; // relative to idbase : opticksdata "export" dir

GPmtLib::GPmtLib(Opticks* ok, GBndLib* bndlib) 
   :
   m_ok(ok),
   m_bndlib(bndlib),
   m_apmt(NULL),
   m_tpmt(NULL)
{
}


GPmtLib* GPmtLib::load(Opticks* ok, GBndLib* blib)
{
    GPmtLib* lib = new GPmtLib(ok, blib);
    lib->loadAnaPmt();
    lib->loadTriPmt();
    lib->dirtyAssociation();
    return lib ; 
}




GPmt* GPmtLib::getLoadedAnalyticPmt()
{
    return m_apmt ; 
}


GMergedMesh* GPmtLib::getPmt()
{
    return m_tpmt ; 
}


void GPmtLib::loadAnaPmt()
{
    NSlice* slice = m_ok->getAnalyticPMTSlice();

    unsigned apmtidx = m_ok->getAnalyticPMTIndex();

    m_apmt = GPmt::load( m_ok, m_bndlib, apmtidx, slice ); 

    LOG(info) << "GPmtLib::loadAnaPmt"
              << " AnalyticPMTIndex " << apmtidx
              << " AnalyticPMTSlice " << ( slice ? slice->description() : "ALL" )
              << " m_apmt " << m_apmt 
              << " Path " << ( m_apmt ? m_apmt->getPath() : "-" ) 
              ;  

    if(m_apmt)
    {
        LOG(verbose) << "GPmtLib::loadAnalyticPmt SUCCEEDED " << m_apmt->getPath()   ; 
        if(m_ok->isDbgAnalytic())
        {
            m_apmt->dump("GPmt::dump --dbganalytic " );
        }
    }    
}


std::string GPmtLib::getTriPmtPath()
{
    OpticksResource* res = m_ok->getResource();
    return res->getBasePath( TRI_PMT_PATH ) ;
}


void GPmtLib::loadTriPmt()
{
    std::string pmtpath = getTriPmtPath(); 

    m_tpmt = GMergedMesh::Load(pmtpath.c_str());

    if(m_tpmt == NULL) LOG(fatal) << "GPmtLib::loadTriPmt FAILED " << pmtpath ;
    assert(m_tpmt);

    m_tpmt->dumpVolumes("GPmtLib::loadTriPmt GMergedMesh::dumpVolumes (before:mmpmt) ");
}


void GPmtLib::dirtyAssociation()
{
    assert( m_apmt && "GPmtLib::dirtyAssociation probably you need option : --apmtload " );
    assert( m_tpmt );

    GParts* pts = m_apmt->getParts();
    m_tpmt->setParts(pts);  
}






