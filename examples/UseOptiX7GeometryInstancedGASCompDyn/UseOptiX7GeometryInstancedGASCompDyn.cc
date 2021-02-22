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

/**
UseOptiX7GeometryInstancedGASCompDyn
======================================

**/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "Ctx.h"
#include "Params.h"
#include "Engine.h"
#include "Geo.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

// Composition::getEyeUVW and examples/UseGeometryShader:getMVP
void getEyeUVW(const glm::vec4& ce, const unsigned width, const unsigned height, glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W )
{
    glm::vec3 tr(ce.x, ce.y, ce.z);  // ce is center-extent of model
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);
    // model frame unit coordinates from/to world 
    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    //glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);

   // View::getTransforms
    glm::vec4 eye_m( -1.f,-1.f,1.f,1.f);  //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.f, 0.f,0.f,1.f); 
    glm::vec4 up_m(   0.f, 0.f,1.f,1.f); 
    glm::vec4 gze_m( look_m - eye_m ) ; 

    const glm::mat4& m2w = model2world ; 
    glm::vec3 eye_ = glm::vec3( m2w * eye_m ) ; 
    //glm::vec3 look = glm::vec3( m2w * look_m ) ; 
    glm::vec3 up = glm::vec3( m2w * up_m ) ; 
    glm::vec3 gaze = glm::vec3( m2w * gze_m ) ;    

    glm::vec3 forward_ax = glm::normalize(gaze);
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,up)); 
    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    float aspect = float(width)/float(height) ;
    float tanYfov = 1.f ;  // reciprocal of camera zoom
    float gazelength = glm::length( gaze ) ;
    float v_half_height = gazelength * tanYfov ;
    float u_half_width  = v_half_height * aspect ;

    U = right_ax * u_half_width ;
    V = top_ax * v_half_height ;
    W = forward_ax * gazelength ; 
    eye = eye_ ; 
}

const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" )
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ptx/"
       << cmake_target
       << "_generated_"
       << cu_stem
       << cu_ext
       << ".ptx" 
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

const char* PPMPath( const char* install_prefix, const char* stem, const char* ext=".ppm" )
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ppm/"
       << stem
       << ext
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

void Params_setView(Params& params, const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_)
{
    params.eye.x = eye_.x ;
    params.eye.y = eye_.y ;
    params.eye.z = eye_.z ;

    params.U.x = U_.x ; 
    params.U.y = U_.y ; 
    params.U.z = U_.z ; 

    params.V.x = V_.x ; 
    params.V.y = V_.y ; 
    params.V.z = V_.z ; 

    params.W.x = W_.x ; 
    params.W.y = W_.y ; 
    params.W.z = W_.z ; 

    params.tmin = tmin_ ; 
    params.tmax = tmax_ ; 
}

void Params_setSize(Params& params, unsigned width_, unsigned height_, unsigned depth_ )
{
    params.width = width_ ;
    params.height = height_ ;
    params.depth = depth_ ;

    params.origin_x = width_ / 2;
    params.origin_y = height_ / 2;
}

int main(int argc, char** argv)
{
    const char* spec = argc > 1 ? argv[1] : "i0" ; 
    std::cout << argv[0] << " spec " << spec << std::endl ;  

    const char* name = "UseOptiX7GeometryInstancedGASCompDyn" ; 
    const char* prefix = getenv("PREFIX"); 
    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );

    const char* cmake_target = name ; 
    const char* ptx_path = PTXPath( prefix, cmake_target, name ) ; 
    std::cout << " ptx_path " << ptx_path << std::endl ; 

    unsigned width = 1024u ; 
    unsigned height = 768u ; 
    unsigned depth = 1u ; 

    Ctx ctx ; 
    Geo geo(spec); 

    float top_extent = geo.getTopExtent() ;  
    glm::vec4 ce(0.f,0.f,0.f, top_extent*1.4f );   // defines the center-extent of the region to view
    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    getEyeUVW( ce, width, height, eye, U, V, W ); 

    std::cout << "main"
              << " top_extent " << top_extent 
              << " tmin " << geo.tmin  
              << " tmax " << geo.tmax
              << std::endl 
              ;  

    Params params ;  
    Params_setView(params, eye, U, V, W, geo.tmin, geo.tmax ); 
    Params_setSize(params, width, height, depth); 

    Engine engine(ptx_path, &params ) ; 
    engine.setGeo(&geo); 

    engine.allocOutputBuffer(); 
    engine.launch(); 
    engine.download(); 

    const char* ppm_path = PPMPath( prefix, name ); 
    std::cout << "write ppm_path " << ppm_path << std::endl ; 
    engine.writePPM(ppm_path);  

    return 0 ; 
}
