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

// opticksgeo/tests/OpticksGeometryTest.cc
// op --topticksgeometry

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksGeometry.hh"

#include "PLOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OKGEO_LOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_COLOR(argc, argv);

    NPY_LOG__ ;
    OKCORE_LOG__ ;
    OKGEO_LOG__ ;
    GGEO_LOG__ ;

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);      // hub calls configure


    return 0 ; 
}
