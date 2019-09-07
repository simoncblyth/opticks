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

//  op --pmt
//  op --pmt --apmtidx 0
//  op --pmt --apmtidx 2 
//  op --pmt --apmtslice 0:10
//

#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"

#include "Opticks.hh"

#include "GBndLib.hh"
#include "GPmt.hh"
#include "GParts.hh"
#include "GCSG.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    for(int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 

    NSlice* slice = ok.getAnalyticPMTSlice();
    unsigned apmtidx = ok.getAnalyticPMTIndex();

    bool constituents = true ; 
    GBndLib* blib = GBndLib::load(&ok, constituents);
    blib->closeConstituents();

    GPmt* pmt = GPmt::load(&ok, blib, apmtidx, slice);

    LOG(info) << argv[0] << " apmtidx " << apmtidx << " pmt " << pmt ; 
    if(!pmt)
    {
        LOG(fatal) << argv[0] << " FAILED TO LOAD PMT " ; 
        return 0 ;
    }

    pmt->dump("GPmt::dump");
  
    return 0 ;
}


