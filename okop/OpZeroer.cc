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

#include <cstddef>

#include "OpZeroer.hh"

#include "OpticksSwitches.h"
#include "Opticks.hh"   
#include "OpticksEvent.hh"   // okc-

// npy-
#include "BTimeKeeper.hh"
#include "PLOG.hh"
#include "NPY.hpp"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrustrap-
#include "TBuf.hh"

// optixrap-
#include "OContext.hh"
#include "OEvent.hh"
#include "OBuf.hh"




OpZeroer::OpZeroer(Opticks* ok, OEvent* oevt)  
   :
     m_ok(ok),
     m_oevt(oevt),
     m_ocontext(oevt->getOContext())
{
}


void OpZeroer::zeroRecords()
{
#ifdef WITH_RECORD
    LOG(info)<<"OpZeroer::zeroRecords" ;

    if( m_ocontext->isInterop() )
    {    
        zeroRecordsViaOpenGL();
    }    
    else if ( m_ocontext->isCompute() )
    {    
        zeroRecordsViaOptiX();
    }    
#endif
}


void OpZeroer::zeroRecordsViaOpenGL()
{
#ifdef WITH_RECORD
    OK_PROFILE("_OpZeroer::zeroRecordsViaOpenGL"); 

    OpticksEvent* evt = m_ok->getEvent();

    NPY<short>* record = evt->getRecordData(); 

    CResource r_rec( record->getBufferId(), CResource::W );

    CBufSpec s_rec = r_rec.mapGLToCUDA<short>() ;

    s_rec.Summary("OpZeroer::zeroRecordsViaOpenGL(CBufSpec)s_rec");

    TBuf trec("trec", s_rec );

    trec.zero();

    r_rec.unmapGLToCUDA();

    OK_PROFILE("OpZeroer::zeroRecordsViaOpenGL"); 
#endif
}


void OpZeroer::zeroRecordsViaOptiX()
{
#ifdef WITH_RECORD
    OK_PROFILE("_OpZeroer::zeroRecordsViaOptiX"); 

    OBuf* record = m_oevt->getRecordBuf() ;

    CBufSpec s_rec = record->bufspec();

    s_rec.Summary("OpZeroer::zeroRecordsViaOptiX(CBufSpec)s_rec");

    TBuf trec("trec", s_rec );

    trec.zero();

    OK_PROFILE("OpZeroer::zeroRecordsViaOptiX"); 
#endif
}


