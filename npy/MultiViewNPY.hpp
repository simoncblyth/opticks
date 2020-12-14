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

#include <vector>
class ViewNPY ; 

#include "NPY_API_EXPORT.hh"

#ifdef _MSC_VER
#pragma warning(push)
// members needs to have dll-interface to be used by clients
#pragma warning( disable : 4251 )
#endif


class NPY_API MultiViewNPY {
    public:
        MultiViewNPY(const char* name="no-name");
        virtual ~MultiViewNPY(); 
        const char* getName();
    public:
        void add(ViewNPY* vec);
        ViewNPY* operator [](const char* name);
        ViewNPY* operator [](unsigned int index);
        unsigned int getNumVecs();


        void Summary(const char* msg="MultiViewNPY::Summary");
        void Print(const char* msg="MultiViewNPY::Print");

    private:
        ViewNPY* find(const char* name);

    private:
        const char*           m_name ; 
        std::vector<ViewNPY*> m_vecs ;  

};


#ifdef _MSC_VER
#pragma warning(pop)
#endif


