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

#include <cassert>

#include "GProperty.hh"
#include "GDomain.hh"
#include "GPropertyMap.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    const char* matname = "FlintGlass" ;
    GPropertyMap<float>* pmap = new GPropertyMap<float>(matname);

    const char* matdir = "$TMP/opticks/GPropertyMapTest";
    pmap->save(matdir);

    GPropertyMap<float>* qmap = GPropertyMap<float>::load(matdir, matname, "material");
    assert(qmap);
    qmap->dump("qmap", 10);


    return 0 ; 
}

