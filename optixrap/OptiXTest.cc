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

#include <iostream>
#include <string>
#include <sstream>

#include <cstdlib>
#include <cstring>

#include "OKConf.hh"
#include "BOpticksResource.hh"
#include "OptiXTest.hh"

#include "PLOG.hh"

const plog::Severity OptiXTest::LEVEL = PLOG::EnvLevel("OptiXTest", "DEBUG") ; 


std::string OptiXTest::ptxname_( const char* projname, const char* name)
{
   std::stringstream ss ; 
   ss << projname << "_generated_" << name << ".ptx" ; 
   return ss.str();
}

const char* OptiXTest::buildptxpath_( const char* cu, const char* buildrel, const char* cmake_target)
{
   std::string ptxname = ptxname_(cmake_target, cu) ; 
   std::string ptxpath = BOpticksResource::BuildProduct(buildrel, ptxname.c_str());

   LOG(LEVEL) 
       << " (BOpticksResource::BuildProduct) "
       << " cu " << cu
       << " buildrel " << buildrel
       << " cmake_target " << cmake_target
       << " ptxname " << ptxname 
       << " ptxpath " << ptxpath
       ; 
       
   const char* cu_name = cu ; 
   const char* ptxrel = buildrel ;  
   std::string path = OKConf::PTXPath( cmake_target, cu, ptxrel ); 

   LOG(LEVEL)
       << " (OKConf::PTXPath) "
       << " cu_name " << cu_name
       << " cmake_target " << cmake_target
       << " ptxrel " << ptxrel
       << " path " << path  
       ; 


   //return strdup(ptxpath.c_str()) ; 
   return strdup(path.c_str()) ; 

   // projdir is needed as build products have paths like
   //      /home/blyth/local/opticks/build/optixrap/tests/tex0Test_generated_tex0Test.cu.ptx 
   //
   // cmake_target
   //      formerly misnamed as projname
   //
   // BUT : what was the reason to get PTX from the build dir ??? rather than install dir 
}

OptiXTest::OptiXTest(optix::Context& context, const char* cu, const char* raygen_name, const char* exception_name, const char* buildrel, const char* cmake_target)
    :
    m_cu(strdup(cu)),
    m_ptxpath(buildptxpath_(cu, buildrel, cmake_target)),
    m_raygen_name(strdup(raygen_name)),
    m_exception_name(strdup(exception_name))
{
    LOG(fatal) << m_ptxpath ; 
    init(context);
}

void OptiXTest::init(optix::Context& context)
{
    LOG(info) << "OptiXTest::init"
              << description()
               ; 

    unsigned num_ray_types = 1; 
    context->setRayTypeCount(num_ray_types);  
    // without setRayTypeCount get SEGV at launch in OptiX_600, changed default or stricter ? an assert would have been nice !
    context->setEntryPointCount( 1 );

    optix::Program raygenProg    = context->createProgramFromPTXFile(m_ptxpath, m_raygen_name);
    optix::Program exceptionProg = context->createProgramFromPTXFile(m_ptxpath, m_exception_name);

    context->setRayGenerationProgram(0,raygenProg);
    context->setExceptionProgram(0,exceptionProg);

    context->setPrintEnabled(true);
    context->setPrintBufferSize(2*2*2*8192);

}

std::string OptiXTest::description()
{
    std::stringstream ss ; 
    ss  
              << " cu " << m_cu
              << " ptxpath " << m_ptxpath
              << " raygen " << m_raygen_name 
              << " exception " << m_exception_name 
              ;

    return ss.str(); 
}

void OptiXTest::Summary(const char* msg)
{
    LOG(info) << msg << description() ;
}


