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
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv) ;
    ok.configure();
    ok.Summary();

    if(argc > 1 && strlen(argv[1]) > 0)
    {
        // canonical usage from opticks-prepare-installcache uses unexpanded argument 
        // '$INSTALLCACHE_DIR/OKC'  expansion is done by BFile.cc
        ok.prepareInstallCache(argv[1]);
    }
    else
    {
        const char* tmp = "$TMP/OKC" ; 
        LOG(info) << "default without argument writes to " << tmp ; 
        ok.prepareInstallCache(tmp);
    }

    return 0 ; 
}


