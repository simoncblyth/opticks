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

#include "BFile.hh"
#include "OptiXTest.hh"
#include "OContext.hh"

#include "SSys.hh"
#include "S_freopen_redirect.hh"

#include "SDirect.hh"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"

const char* TMPDIR = "$TMP/optixrap/redirectLogTest" ; 

int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    OContext::SetupOptiXCachePathEnvvar(); 
    optix::Context context = optix::Context::create();

    //const char* buildrel = "optixrap/tests" ; 
    const char* ptxrel = "tests" ; 
    const char* cmake_target = "redirectLogTest" ; 
    OptiXTest* test = new OptiXTest(context, "redirectLogTest.cu", "minimal", "exception", ptxrel, cmake_target ) ;
    test->Summary(argv[0]);

    //unsigned width = 512 ; 
    //unsigned height = 512 ; 

    unsigned width = 16 ; 
    unsigned height = 16 ; 



    // optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);

    context->validate();
    context->compile();


/*
    // hmm this fails to capture anything
    // lower level freopen redirect works 

    std::stringstream coutbuf ;
    std::stringstream cerrbuf ;
    {    
         cout_redirect out_(coutbuf.rdbuf());
         cerr_redirect err_(cerrbuf.rdbuf()); 

         context->launch(0, width, height);
    }    
    std::string out = coutbuf.str();
    std::string err = cerrbuf.str();


    LOG(info) << " captured out " << out.size() << " err " << err.size() ; 
    LOG(info) << "out("<< out.size() << "):\n" << out ; 
    LOG(info) << "err("<< err.size() << "):\n" << err ; 
*/



    std::string p = BFile::FormPath(TMPDIR, "redirectLogTest.log" ); 
    const char* path = p.c_str(); 

    SSys::Dump(path);
    {
        S_freopen_redirect sfr(stdout, path );
        context->launch(0, width, height);
        SSys::Dump(path);
    }
    SSys::Dump(path);



    NPY<float>* npy = NPY<float>::make(width, height, 4) ;
    npy->zero();

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 

 
    npy->save(TMPDIR, "redirectLogTest.npy");


    return 0;
}
