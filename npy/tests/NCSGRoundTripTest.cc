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

// TEST=NCSGRoundTripTest om-t

#include "NPY.hpp"
#include "NConvexPolyhedron.hpp"

#include "OPTICKS_LOG.hh"
#include "NCSG.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    nconvexpolyhedron* a_cpol = nconvexpolyhedron::make_trapezoid_cube() ; 
    assert( a_cpol ) ; 
    a_cpol->dump_planes(); 
    a_cpol->dump_srcvertsfaces(); 

    

    a_cpol->verbosity = 2 ; 

    LOG(error) << " before Adopt " ; 
    NCSG* a = NCSG::Adopt(a_cpol) ; 
    LOG(error) << " after Adopt " ; 

    NPY<float>* a_planes = a->getPlaneBuffer();     
    LOG(info) << " a_planes " << a_planes->getShapeString() ; 

    const char* treedir = "$TMP/NPY/NCSGRoundTripTest/nconvexpolyhedron/1" ;
    a->savesrc(treedir);     

    NCSG* b = NCSG::Load(treedir) ;     

    NPY<float>* b_planes = b->getPlaneBuffer();     
    LOG(info) << " b_planes " << b_planes->getShapeString() ; 


    nconvexpolyhedron* b_cpol = dynamic_cast<nconvexpolyhedron*>(b->getRoot());     
    assert( b_cpol ) ; 
    b_cpol->dump_planes();  
    b_cpol->dump_srcvertsfaces(); 


    return 0 ; 
}
