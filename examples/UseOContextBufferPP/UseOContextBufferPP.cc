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

/**
UseOContextBufferPP
===================

Higher level variant of UseOptiXBufferPP

**/


#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include "OPTICKS_LOG.hh"
#include "OKConf.hh"
#include "NPY.hpp"
#include "SSys.hh"
#include "OContext.hh"
#include "OpticksBufferControl.hh"
#include "Opticks.hh"


void printUsageAndExit(const char* name)
{
    std::cout 
        << "Expect either zero, one or two arguments eg::" << std::endl 
        << std::endl 
        << "    " << name << std::endl 
        << "    " << name << " " << "bufferTest.cu            ## progname defaults to bufferTest here " << std::endl 
        << "    " << name << " " << "bufferTest.cu printTest  ## progname specified as printTest " << std::endl 
        << std::endl 
        ;
    exit(1); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    Opticks ok(argc, argv, "--compute --printenabled" ); 
    ok.configure();


    const char* cmake_target = "UseOContextBufferPP" ; 
    const char* cu_name = NULL ;  
    const char* progname = NULL  ;

    if((argc > 1 && ( strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 )) || argc > 3) printUsageAndExit(argv[0]);  

    if(argc == 1)
    {
        cu_name = "bufferTest.cu" ;  // only needs to be unique within context of the cmake_target, not the whole of Opticks
        progname = "bufferTest" ;  
    } 
    else if( argc == 2)
    {
        std::string arg = argv[1] ;
        std::string base = arg.substr(0, arg.find_last_of(".")) ; 
        cu_name = strdup( arg.c_str() ); 
        progname = strdup( base.c_str() ) ; 
    }
    else if( argc == 3)
    {
        cu_name = argv[1] ; 
        progname = argv[2] ; 
    }


    const char* ptx_path = OKConf::PTXPath( cmake_target, cu_name ); 

    for(int i=0 ; i < argc ; i++) std::cout << argv[i] << " " ;  
    std::cout 
        << std::endl
        << " cmake_target : " << cmake_target << std::endl
        << " cu_name (1)  : " << cu_name << std::endl 
        << " progname (2) : " << progname << std::endl 
        << " ptx_path     : " << ptx_path << std::endl 
        ;



    unsigned size = 10u ; 
    NPY<float>* in_npy = NPY<float>::make(size,1,4) ;
    in_npy->fill(42.f);

    NPY<float>* out_npy = NPY<float>::make(size,1,4) ;
    out_npy->fill(0.f);


    OpticksBufferControl::Add( in_npy->getBufferControlPtr(), "OPTIX_INPUT_ONLY" );
    OpticksBufferControl::Add( out_npy->getBufferControlPtr(), "OPTIX_OUTPUT_ONLY,COMPUTE_MODE" );  // COMPUTE_MODE needed to effect the download : poor name ?



    OContext* ctx = OContext::Create(&ok, cmake_target );
    optix::Context context = ctx->getContext();


/*
    context->setRayTypeCount(1); 
    context->setPrintEnabled(true); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
*/


   /*
    optix::Program program = context->createProgramFromPTXFile( ptx_path , progname );  
    unsigned entry_point_index = 0u ;
    context->setRayGenerationProgram( entry_point_index, program ); 
   */

   bool defer = true ; 
   unsigned entry_point_index = ctx->addEntry( cu_name, progname, "exception", defer ); 
    

   if(defer)
   {
       ctx->close();   
   }
   else
   {
       context->setEntryPointCount(1); 
   }



    // create and configure in_buffer

    optix::Buffer in_buffer = ctx->createBuffer<float>( in_npy, "in_buffer");
   /*
    unsigned int in_type =  RT_BUFFER_INPUT ; 
    RTformat in_format = RT_FORMAT_FLOAT4 ;
    optix::Buffer in_buffer = context->createBuffer(in_type);
    in_buffer->setFormat( in_format ) ; 
    in_buffer->setSize( size ) ;   // number of quads
  */ 
    context["in_buffer"]->set( in_buffer );

    // create and configure out_buffer

   /* 
    unsigned int out_type =  RT_BUFFER_OUTPUT ; 
    RTformat out_format = RT_FORMAT_FLOAT4 ;
    optix::Buffer out_buffer = context->createBuffer(out_type);
    out_buffer->setFormat( out_format ) ; 
    out_buffer->setSize( size ) ;   // number of quads
   */

    optix::Buffer out_buffer = ctx->createBuffer<float>( out_npy, "out_buffer");
    context["out_buffer"]->set( out_buffer );


    // upload in_buffer
    /*
    unsigned numBytes = in_npy->getNumBytes(0) ;
    memcpy( in_buffer->map(), in_npy->getBytes(), numBytes );
    in_buffer->unmap() ; 
    */
    ctx->upload<float>(in_buffer, in_npy) ; 



    // launch the kernel, that just copies from in to out 
    unsigned width = size ; 
    //context->launch( entry_point_index , width  ); 

    ctx->launch( OContext::LAUNCH, entry_point_index,  width, 1, NULL );




    // download out_buffer
  /*
    void* out_ptr = out_buffer->map() ;
    out_npy->read( out_ptr );
    out_buffer->unmap();
  */
    ctx->download<float>( out_buffer, out_npy ); 


    const char* out_path = "$TMP/UseOContextBufferPP/out.npy"; 
    out_npy->save(out_path);
    SSys::npdump(out_path, "np.float32");

    LOG(info) << argv[0] ; 

    return 0 ; 
}


