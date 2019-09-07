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

#include "NQuad.hpp"
#include "NBBox.hpp"
#include <vector>

#include "NPY_API_EXPORT.hh"

struct nnode ; 
class NTrianglesNPY ; 


class NPY_API NMarchingCubesNPY {
    public:
        NMarchingCubesNPY(int nx, int ny=0, int nz=0);

        NTrianglesNPY* operator()(nnode* node); 
    private:
         void march(nnode* node);
         NTrianglesNPY* makeTriangles();
    private:
        int m_nx ; 
        int m_ny ; 
        int m_nz ; 

        double m_isovalue ; 
        double m_scale ; 
        double m_lower[3] ;
        double m_upper[3] ;

        std::vector<double> m_vertices ; 
        std::vector<size_t> m_polygons ; 

    //    ntrange3<double>    m_range ; 


};


