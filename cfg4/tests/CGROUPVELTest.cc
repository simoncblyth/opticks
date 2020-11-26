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


#include "NPY.hpp"
#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksMode.hh"
#include "CMaterialLib.hh"
#include "CFG4_BODY.hh"
#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv);

    OpticksHub hub(&ok); 

    CMaterialLib* clib = new CMaterialLib(&hub); 

    LOG(info) << argv[0] << " convert " ; 

    clib->convert();

    LOG(info) << argv[0] << " dump " ; 

    clib->dump();

    clib->saveGROUPVEL("$TMP/cfg4/CGROUPVELTest");



    return 0 ; 
}
