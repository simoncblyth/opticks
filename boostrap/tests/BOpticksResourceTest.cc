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

#include "SSys.hh"
#include "BResource.hh"
#include "BOpticksResource.hh"
#include "OPTICKS_LOG.hh"


#ifdef OLD_RESOURCE
struct BOpticksResourceTest
{
    BOpticksResourceTest(const char* idpath)
        :
        _res()
    {
        _res.setupViaID(idpath);
        _res.Summary();
    }
    
    BOpticksResourceTest(const char* srcpath, const char* srcdigest)
        :
        _res()
    {
        _res.setupViaSrc(srcpath, srcdigest);
        _res.Summary();
    }

    BOpticksResource _res ; 

};




void test_ViaSrc()
{
    const char* srcpath   = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    const char* srcdigest  = SSys::getenvvar("DEBUG_OPTICKS_SRCDIGEST", "0123456789abcdef0123456789abcdef" );

    assert( srcpath && srcdigest );
 
    BOpticksResourceTest brt(srcpath, srcdigest) ; 
    BOpticksResourceTest brt2(brt._res.getIdPath()) ; 
}


void test_Setup()
{
    const char* idpath  = SSys::getenvvar("IDPATH");
    if(!idpath) return  ;     

    LOG(info) << " starting from IDPATH " << idpath ; 
    BOpticksResourceTest brt(idpath) ; 
    BOpticksResourceTest brt2(brt._res.getSrcPath(), brt._res.getSrcDigest()) ; 

    // the two setup approaches, should yield exactly the same paths 
    BResource::Dump("BOpticksResourceTest"); 
}
#endif


void test_IsGeant4EnvironmentDetected()
{
    bool detect = BOpticksResource::IsGeant4EnvironmentDetected()  ; 
    LOG(info) << detect ; 
}

void test_GetCachePath()
{
    const char* path = BOpticksResource::GetCachePath("GNodeLib/all_volume_identity.npy"); 
    LOG(info) << path ; 
}

void test_getGDMLAuxTargetLVName()
{
    BOpticksResource* rsc = BOpticksResource::Get(NULL) ;  // use preexisting instance or create new one
    const char* target = rsc->getGDMLAuxTargetLVName() ; 
    LOG(info) << "getGDMLAuxTargetLVName : " << target ; 
    const char* keyspec= rsc->getKeySpec() ;
    LOG(info) << "getKeySpec : " << keyspec ; 

    LOG(info) << std::endl << rsc->export_(); 


}  



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
#ifdef OLD_RESOURCE
    //test_ViaSrc(); 
    //test_Setup(); 
#endif

 
    //test_IsGeant4EnvironmentDetected(); 

    //test_GetCachePath(); 

    test_getGDMLAuxTargetLVName(); 

    return 0 ; 
}
// om-;TEST=BOpticksResourceTest om-t 

