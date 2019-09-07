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


#include "OPTICKS_LOG.hh"

// npy-
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

// oglrap-
#include "AxisApp.hh"

// optixrap-
#include "Opticks.hh"
#include "OContext.hh"

// opticksgl-
#include "OAxisTest.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv, "--interop");
    ok.configure();

    LOG(info) << argv[0] ; 

    AxisApp axa(&ok); 
    NPY<float>* npy = axa.getAxisData();
    assert(npy->hasShape(3,3,4));

    /*
    MultiViewNPY* mvn = aa.getAxisAttr();
    ViewNPY* vpos = (*mvn)["vpos"];
    NPYBase* npyb = vpos->getNPY();  // NB same npy holds vpos, vdir, vcol
    assert(npy == npyb);
    */
   
    //OContext::Mode_t mode = OContext::INTEROP ;
    optix::Context context = optix::Context::create();

    OContext* m_ocontext = new OContext(context, &ok, false );

    OAxisTest* oat = new OAxisTest(m_ocontext, npy);
    oat->prelaunch();

    axa.setLauncher(oat);
    axa.renderLoop();

    return 0 ; 
}

// note flakiness, sometimes the axis appears sometimes not


