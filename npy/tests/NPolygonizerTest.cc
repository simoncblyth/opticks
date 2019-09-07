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
#include "NCSG.hpp"
#include "NPolygonizer.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    const char* treedir = argc > 1 ? argv[1] : "$TMP/tboolean-hyctrl--/1" ;

    const char* config = "csg_bbox_parsurf=1" ;

    NCSG* csg = NCSG::Load(treedir, config );

    if(!csg)
    {
        LOG(fatal) << "NO treedir/tree " << treedir ; 
        return 0 ;  
    }

    NPolygonizer poly(csg);

    NTrianglesNPY* tris = poly.polygonize();

    assert(tris);

    return 0 ; 
}


