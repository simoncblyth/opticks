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

// ok/tests/OKTest.cc
#include "OKMgr.hh"

#include "PLOG.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"
#include "MESHRAP_LOG.hh"
#include "OKGEO_LOG.hh"
#include "OGLRAP_LOG.hh"

#ifdef OPTICKS_OPTIX
#include "CUDARAP_LOG.hh"
#include "THRAP_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"
#include "OKGL_LOG.hh"
#endif

#include "OK_LOG.hh"

/**

OKTest
================
**/

int main(int argc, char** argv)
{
    //PLOG_(argc, argv);
    PLOG_COLOR(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
    ASIRAP_LOG__ ;
    MESHRAP_LOG__ ;
    OKGEO_LOG__ ;
    OGLRAP_LOG__ ;

#ifdef OPTICKS_OPTIX
    CUDARAP_LOG__ ;
    THRAP_LOG__ ;
    OXRAP_LOG__ ;
    OKOP_LOG__ ;
    OKGL_LOG__ ;
#endif

    OK_LOG__ ;


 
    OKMgr ok(argc, argv);
    ok.propagate();
    ok.visualize();

    return ok.rc();
}

