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
#include "BOpticksResource.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"

struct BOpticksResourceTest
{
    BOpticksResourceTest(const char* idpath)
        :
        _res(false)
    {
        _res.setupViaID(idpath);
        _res.Summary();
    }
    
    BOpticksResourceTest(const char* srcpath, const char* srcdigest)
        :
        _res(false)
    {
        _res.setupViaSrc(srcpath, srcdigest);
        _res.Summary();
    }

    BOpticksResource _res ; 

};



/*

    const char* treedir_ = brt._res.getDebuggingTreedir(argc, argv);  //  requires the debugging only IDPATH envvar
    std::string treedir = treedir_ ? treedir_ : "/tmp/error-no-IDPATH-envvar" ; 

    std::cout 
              << " treedir " << treedir
              << std::endl 
              ;

*/



void test_ViaSrc()
{
    const char* srcpath   = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    const char* srcdigest  = SSys::getenvvar("DEBUG_OPTICKS_SRCDIGEST", "0123456789abcdef0123456789abcdef");

    assert( srcpath && srcdigest );
 
    BOpticksResourceTest brt(srcpath, srcdigest) ; 
    BOpticksResourceTest brt2(brt._res.getIdPath()) ; 
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ; 

    const char* idpath  = SSys::getenvvar("IDPATH");
    if(!idpath) return 0 ;     

    LOG(info) << " starting from IDPATH " << idpath ; 
    BOpticksResourceTest brt(idpath) ; 
    BOpticksResourceTest brt2(brt._res.getSrcPath(), brt._res.getSrcDigest()) ; 

    // the two setup approaches, should yield exactly the same paths 

    return 0 ; 
}
