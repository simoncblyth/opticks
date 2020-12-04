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

#include "PLOG.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#ifdef  WITH_OPENMESHRAP
#include "MESHRAP_LOG.hh"
#endif
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


int main(int argc, char** argv)
{
    //PLOG_(argc, argv);
    PLOG_COLOR(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
#ifdef  WITH_OPENMESHRAP
    MESHRAP_LOG__ ;
#endif
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


    const char* msg = argv[0] ;

    SYSRAP_LOG::Check(msg) ;
    BRAP_LOG::Check(msg) ;
    NPY_LOG::Check(msg) ;
    OKCORE_LOG::Check(msg) ;
    GGEO_LOG::Check(msg) ;
#ifdef  WITH_OPENMESHRAP
    MESHRAP_LOG::Check(msg) ;
#endif
    OKGEO_LOG::Check(msg) ;
    OGLRAP_LOG::Check(msg) ;

#ifdef OPTICKS_OPTIX
    CUDARAP_LOG::Check(msg) ;
    THRAP_LOG::Check(msg) ;
    OXRAP_LOG::Check(msg) ;
    OKOP_LOG::Check(msg) ;
    OKGL_LOG::Check(msg) ;
#endif
    OK_LOG::Check(msg) ;


    return 0 ;
} 

/*

   Seems cannot turn up the loglevel in the projects, all are stuck at the fatal set in main.
   (chaining effect >?)

   LogTest --fatal --asirap trace
   LogTest --fatal --okop warn 
 
   However the converse does work. Can turn down project log level, but only down so far as the primary level.

   LogTest --trace --okop fatal --ggv fatal --okgl fatal --oxrap fatal --thrap fatal --cudarap fatal --oglrap fatal --okgeo fatal --meshrap fatal --asirap fatal --ggeo fatal --okcore fatal --npy fatal --sysrap fatal --brap fatal


   Generally there is very little need for logging control in the main, so can leave that 
   at trace and default the projects to info



*/

