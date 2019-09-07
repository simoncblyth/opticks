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

#include <cstring>
#include <string>
#include <vector>

// vi Composition.hh Scene.hh Trackball.hh Camera.hh View.hh Clipper.hh
// vi Composition.cc Scene.cc Trackball.cc Camera.cc View.cc Clipper.cc

/**
NConfigurable
==============

::

    find . \( -name '*.hh' -or -name '*.hpp' \) -exec grep -l NConfigurable {} \;  
    ./opticksgeo/OpticksHub.hh
    ./ggeo/GGeo.hh
    ./optickscore/Camera.hh
    ./optickscore/View.hh
    ./optickscore/Demo.hh
    ./optickscore/Bookmarks.hh
    ./optickscore/Composition.hh
    ./optickscore/Clipper.hh
    ./optickscore/Trackball.hh
    ./npy/NConfigurable.hpp
    ./npy/NState.hpp
    ./oglrap/OpticksViz.hh
    ./oglrap/Scene.hh

    vi `!!`

**/


#include "NPY_API_EXPORT.hh"

class NPY_API NConfigurable {
    public: 
       virtual const char* getPrefix() = 0 ; 
       virtual void configure(const char* name, const char* value) = 0;
       virtual std::vector<std::string> getTags() = 0;
       virtual std::string get(const char* name) = 0 ;
       virtual void set(const char* name, std::string& val) = 0 ;

//       virtual void command(const char* ctrl) ; 

};



