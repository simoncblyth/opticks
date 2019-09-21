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
#include <string>
#include <sstream>

#include "ORng.hh"
#include "OConfig.hh"
#include "OContext.hh"
#include "Opticks.hh"

#include "OPTICKS_LOG.hh"

/**
ORngTest
============

::

   ORngTest --printenabled

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks ok(argc, argv, "--compute");
    ok.configure();

    unsigned version = OConfig::OptiXVersion()  ;
    LOG(info) << argv[0] << " OPTIX_VERSION " << version ; 

    OContext* ctx = OContext::Create(&ok, "ORngTest", "tests");
    optix::Context context = ctx->getContext();
    int entry = ctx->addEntry("ORngTest.cu", "ORngTest", "exception");
    new ORng(&ok, ctx);  

    unsigned size = 1000 ; 
    ctx->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, NULL );
    ctx->launch( OContext::LAUNCH, entry,  size, 1,  NULL );

    delete ctx ; 

    return 0 ;     
}


