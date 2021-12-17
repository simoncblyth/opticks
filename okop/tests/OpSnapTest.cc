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

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "OpMgr.hh"

/**
OpSnapTest (formerly OpTest)
=============================

Loads geometry from cache, creates sequence of ppm raytrace snapshots of geometry::

    OpSnapTest 
        triangulated geometry default

    OpSnapTest --gltf 3
        FAILS : the old default geocache has some issues 

    OPTICKS_RESOURCE_LAYOUT=104 OpSnapTest --gltf 3
        succeeds with the ab- validated geocache : creating analytic raytrace snapshots

Example commandlines::

    OpSnapTest --snapconfig steps=0,postfix=.jpg 
    OpSnapTest --snapconfig steps=0,postfix=.png
    OpSnapTest --snapconfig steps=0,postfix=.ppm

    OpSnapTest --snapconfig steps=0,postfix=.jpg --targetpvn pAcrylic

**/


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    Opticks ok(argc, argv, "--tracer");   // tempted to put --embedded here 
    OpMgr op(&ok);

    int rc = op.render_snap();
    if(rc) LOG(fatal) << " rc " << rc ; 

    return 0 ; 
}





