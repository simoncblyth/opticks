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

#include "BOpticksEvent.hh"
#include "BOpticksResource.hh"

#include "NLoad.hpp"
#include "NPY.hpp"
#include "PLOG.hh"

const plog::Severity NLoad::LEVEL = info ; 

std::string NLoad::GenstepsPath(const char* det, const char* typ, const char* tag)
{
    const char* gensteps_dir = BOpticksResource::GenstepsDir();  // eg /usr/local/opticks/opticksdata/gensteps
    BOpticksEvent::SetOverrideEventBase(gensteps_dir) ;
    BOpticksEvent::SetLayoutVersion(1) ;     

    LOG(LEVEL) 
         << " gensteps_dir " << gensteps_dir ; 
         ; 

    const char* pfx = NULL ; 
    const char* stem = NULL ; 
    std::string path = BOpticksEvent::path(pfx, det, typ, tag, stem, ".npy");

    BOpticksEvent::SetOverrideEventBase(NULL) ;
    BOpticksEvent::SetLayoutVersionDefault() ;

    return path ; 
}

NPY<float>* NLoad::Gensteps(const char* det, const char* typ, const char* tag)
{
    std::string path = GenstepsPath(det, typ, tag);
    NPY<float>* gs = NPY<float>::load(path.c_str()) ;
    return gs ; 
}


std::string NLoad::directory(const char* pfx, const char* det, const char* typ, const char* tag, const char* anno)
{
   std::string tagdir = BOpticksEvent::directory(pfx, det, typ, tag, anno ? anno : NULL );  
   return tagdir ; 
}

std::string NLoad::reldir(const char* pfx, const char* det, const char* typ, const char* tag )
{
   std::string rdir = BOpticksEvent::reldir(pfx, det, typ, tag );  
   return rdir ; 
}


