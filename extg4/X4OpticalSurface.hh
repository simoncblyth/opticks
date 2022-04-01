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

#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <sstream>

#include "X4_API_EXPORT.hh"

#ifdef WITH_PLOG
#include "plog/Severity.h"
#endif

#include "X4OpticalSurfaceModel.hh"
#include "X4OpticalSurfaceFinish.hh"
#include "X4SurfaceType.hh"

class G4OpticalSurface ; 
class GOpticalSurface ; 

/**
X4OpticalSurface
==================

CAUTION : Only a small fraction of Geant4 optical surface handling 
has been ported to Opticks.

See also:

X4OpticalSurfaceModel 
    string consts for G4OpticalSurfaceModel enum  { glisur, unified, ... }

X4OpticalSurfaceFinish
    string consts for G4OpticalSurfaceFinish enum { polished, polishedfrontpainted, ... }

X4SurfaceType
    string consts for G4SurfaceType enum { dielectric_metal, dielectric_dielectric, ... }

**/

struct  X4_API X4OpticalSurface 
{
        const char* name ; 
        const char* model ; 
        const char* finish ; 
        const char* type ; 
        const char* value ; 

#ifdef WITH_PLOG
        static const plog::Severity LEVEL ; 
#endif
        static X4OpticalSurface* FromString(const char* spec, char delim=','); 
        static GOpticalSurface* Convert(const G4OpticalSurface* const src );

};


/**
X4OpticalSurface::FromString
-------------------------------

"surfname,unified,polished,dielectric_dielectric,1.0" 

**/

inline X4OpticalSurface* X4OpticalSurface::FromString(const char* spec, char delim)
{
    std::vector<std::string> elem ; 
    std::stringstream ss; 
    ss.str(spec)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 

    assert( elem.size() == 5 ); 

    X4OpticalSurface* xsurf = new X4OpticalSurface ; 

    xsurf->name   = strdup(elem[0].c_str()); 
    xsurf->model  = strdup(elem[1].c_str()); 
    xsurf->finish = strdup(elem[2].c_str()); 
    xsurf->type   = strdup(elem[3].c_str()); 
    xsurf->value  = strdup(elem[4].c_str()); 

    return xsurf ; 
}


