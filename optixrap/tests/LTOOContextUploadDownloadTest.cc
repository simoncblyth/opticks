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

#include "Opticks.hh"

#include "OptiXTest.hh"
#include "OContext.hh"

#include "NPY.hpp"
#include "DummyGenstepsNPY.hpp"

#include "OBuf.hh"

#include "BOpticksEvent.hh"
#include "BOpticksResource.hh"

#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"
#include "BOpticksResource.hh"

#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv, "--compute" );
    ok->configure();

    //NPY<float>* npy = NLoad::Gensteps("juno", "cerenkov", "1") ; 
    NPY<float>* npy = DummyGenstepsNPY::Make(100) ; 

    assert(npy);
    std::string path = npy->getMeta<std::string>("path", ""); 
    LOG(info) << " path " << path ; 

    npy->dump("NPY::dump::before", 2);

    // manual buffer control, normally done via spec in okc-/OpticksEvent 
    npy->setBufferControl(OpticksBufferControl::Parse("OPTIX_INPUT_OUTPUT"));


    OContext* ctx = OContext::Create( ok );
    optix::Context context = ctx->getContext(); 

    unsigned entry = ctx->addEntry("LTminimalTest.cu", "minimal", "exception");


    optix::Buffer buffer = ctx->createBuffer<float>( npy, "demo");
    context["output_buffer"]->set(buffer);
    OBuf* genstep_buf = new OBuf("genstep", buffer);

    OContext::upload(buffer, npy);

    genstep_buf->dump<unsigned int>("LT::OBuf test: ", 6*4, 3, 6*4*10);
    LOG(info) << "check OBuf begin.";
    // LT: check OBuf
    npy->zero();
    genstep_buf->download(npy);
    npy->dump("NPY::dump::after", 2);
    LOG(info) << "check OBuf end.";

    unsigned ni = 10 ; 
    ctx->launch( OContext::VALIDATE,  entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after VALIDATE: ", 6*4, 3, 6*4*10);
    ctx->launch( OContext::COMPILE,   entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after COMPILE: ", 6*4, 3, 6*4*10);
    ctx->launch( OContext::PRELAUNCH, entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after PRELAUNCH: ", 6*4, 3, 6*4*10);
    ctx->launch( OContext::LAUNCH,    entry, ni, 1);
    genstep_buf->dump<unsigned int>("LT::OBuf test after LAUNCH: ", 6*4, 3, 6*4*10);

    npy->zero();

    OContext::download( buffer, npy );

    NPYBase::setGlobalVerbose();

    // npy->dump();
    npy->save("$TMP/optixrap/LTOOContextUploadDownloadTest/OOContextUploadDownloadTest_1.npy");

    delete ctx ; 

    return 0;
}
