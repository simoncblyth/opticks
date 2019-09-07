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

#include "N.hpp"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"
#include "NSDF.hpp"
#include "NNode.hpp"
#include "NNodePoints.hpp"
#include "NCSG.hpp"


N::N(nnode* node, const nmat4triple* transform, const NSceneConfig* config, float surface_epsilon ) 
    : 
         node(node), 
         transform(transform),
         config(config),
         points(new NNodePoints(node, config)),
         nsdf(node->sdf(), transform->v)
{
         points->setEpsilon(surface_epsilon);

         tots = points->collect_surface_points() ;

         const std::vector<glm::vec3>& src = points->getCompositePoints();
         for(unsigned i=0 ; i < src.size() ; i++)  model.push_back(src[i] ) ; 

        // huh there is a local vector of vec3 named model ?

         // populate local vector of vec3 with transformed points from src 
         float w = 1.0f ; 
         transform->apply_transform_t( local, model, w );

         num = model.size();
         assert( local.size() == num );

} 

glm::uvec4 N::classify(const std::vector<glm::vec3>& qq, float epsilon, unsigned expect, bool dump )
{
      nsdf.classify(qq, epsilon, expect, dump);
      return nsdf.tot ; 
}

std::string N::desc() const 
{
      std::stringstream ss ;  
      ss
         << ( node->label ? node->label : "-" )
         << " nsdf: " << nsdf.desc() 
         << nsdf.detail()
         ; 
     return ss.str();
}




void N::dump_points(const char* msg)
{
     std::cout << " local points are model points transformed with transform->t (the placing transform) " << std::endl ;  
     std::cout << msg 
               << std::endl 
               << gpresent( "t", transform->t )
               << std::endl 
               ;

      // NB the length of local and model will not typically be the same as the queried points
      for(unsigned i=0 ; i < num ; i++) 
             std::cout 
               << " model " << gpresent( model[i] ) 
               << " local " << gpresent( local[i] ) 
               << std::endl 
               ;

}


