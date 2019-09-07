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


#include "SSys.hh"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NNode.hpp"
#include "NNodePoints.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"
#include "NBox.hpp"
#include "Nuv.hpp"

#include "OPTICKS_LOG.hh"


void test_collect_surface_points()
{
    glm::vec3 tlate(5,5,5);
    glm::vec4 aa(0,0,0,10);
    glm::vec4 bb(0,0,0,10);

    nbox* a = make_box(aa.x,aa.y,aa.z,aa.w);
    nbox* b = make_box(bb.x,bb.y,bb.z,bb.w);
    b->transform = nmat4triple::make_translate( tlate );    

    nintersection* ab = nintersection::make_intersection(a, b); 

    a->parent = ab ;  // parent hookup usually done by NCSG::import_r 
    b->parent = ab ;   
    ab->update_gtransforms();  // recurse over tree using parent links to set gtransforms

    ab->dump();

    unsigned verbosity = SSys::getenvint("VERBOSITY", 1) ;

    ab->verbosity = verbosity ; 

    NNodePoints pts(ab, NULL );
    glm::uvec4 tot = pts.collect_surface_points();

    pts.dump("test_collect_surface_points.pts", verbosity > 4 ? 1000 : 20  );

    std::cout << "intersection:  (inside/surface/outside/select)  " << glm::to_string(tot) << std::endl ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_collect_surface_points();

    return 0 ; 
}



