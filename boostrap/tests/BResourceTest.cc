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

// TEST=BResourceTest om-t 

#include "OPTICKS_LOG.hh"
#include "BOpticksResource.hh"
#include "BResource.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    BOpticksResource br ; 

    const char* key = argc > 1 ? argv[1] : "tmpuser_dir" ; 
    const char* nval = argc > 2 ? argv[2] : "/tmp" ; 
    const char* val = BResource::GetDir(key) ; 

    LOG(info) 
        << " key " << key 
        << " val " << val
        << " nval " << nval
        ; 


    BResource::Dump("BResourceTest.0"); 
    BResource::SetDir(key, nval) ; 
    BResource::Dump("BResourceTest.1"); 

    return 0 ; 
}
