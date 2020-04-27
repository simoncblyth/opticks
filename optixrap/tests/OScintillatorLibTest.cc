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


#include "OScintillatorLib.hh"
#include "Opticks.hh"
#include "OContext.hh"
#include "GScintillatorLib.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    // Sajan reports that with some unreported versions of OptiX+CUDA+Driver 
    // this alone fails to init giving "GPU not found" error 
    // OContext::SetupOptiXCachePathEnvvar(); 
    // optix::Context context = optix::Context::create();

    const char* cmake_target = "OScintillatorLibTest"  ;
    const char* ptxrel = "tests" ; 
    OContext* ctx = OContext::Create(&ok, cmake_target, ptxrel);
    optix::Context context = ctx->getContext();
 


    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    const char* slice = "0:1" ; 
    oscin->convert(slice);

    LOG(info) << "DONE"  ;

    return 0 ; 
}


