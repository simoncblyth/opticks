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


#include <vector>

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"
#include "CCtx.hh"
#include "CRecState.hh"
#include "CPhoton.hh"



void test_CPhoton(CPhoton& p, const std::vector<unsigned>& flags )
{
    for(unsigned i=0 ; i < flags.size() ; i++)
    {
        unsigned flag = flags[i];
        unsigned material = i + 1 ; // placeholder
 
        p.add( flag, material );
        p.increment_slot();

        LOG(info) << p.desc() ; 
    }
}

void test_CPhoton_copy( CPhoton& p )
{
    CPhoton c(p);

    LOG(info) << "p:" << p.desc() ; 
    LOG(info) << "c:" << c.desc() ; 

}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv );
    ok.configure();
    ok.setSpaceDomain(0,0,0,0); // for configureDomains

    OpticksEvent* evt = ok.makeEvent(false, 0u);

    CCtx ctx(&ok);
    ctx.initEvent(evt);

    CRecState s(ctx);
    CPhoton   p(ctx, s) ; 

    {
        std::vector<unsigned> flags ; 
        flags.push_back(TORCH);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_REFLECT);
        flags.push_back(BULK_ABSORB);

        p.clear();
        s.clear();

        test_CPhoton(p, flags );
        test_CPhoton_copy(p);

    }



    {
        std::vector<unsigned> flags ; 
        flags.push_back(TORCH);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_REFLECT);
        flags.push_back(BULK_SCATTER);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(SURFACE_DREFLECT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(SURFACE_SREFLECT);
        flags.push_back(BULK_ABSORB);


        p.clear();
        s.clear();

        test_CPhoton(p, flags );
        test_CPhoton_copy(p);
    }


    return 0 ; 
}
 
