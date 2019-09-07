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

#include <iostream>

#include "G4Point3D.hh"
#include "X4RotationMatrix.hh"
#include "X4Transform3D.hh"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"

#include "OPTICKS_LOG.hh"


void test_convert()
{
    // create glm::mat4 transform

    glm::vec3 tlat(10,1,3);
    glm::vec4 axis_angle(1,1,1,45.); 
    //glm::vec4 axis_angle(0,0,1,45.); 
    glm::vec3 scal(1,1,1);
    glm::mat4 trs = nglmext::make_transform("trs", tlat, axis_angle, scal );

    std::cout << gpresent("trs", trs) << std::endl ; 

    // apply the transform to a single point 

    glm::vec4 p(1,0,0,1);
    glm::vec4 pt = trs * p ; 

    std::cout 
           << gpresent("trs", trs) 
           << std::endl
           << gpresent("p", p) 
           << std::endl
           << gpresent("pt", pt) 
           ; 

    // convert glm ->  G4 

    G4Transform3D t0 = X4Transform3D::Convert(trs);

    G4Point3D pp(p.x, p.y, p.z ) ; 
    G4Point3D ppt = t0 * pp ;  

    std::cout 
          << "pp" << pp << std::endl
          << "ppt" << ppt << std::endl
          ; 

    // compare the result of the G4 performed transform with the glm result

    float eps = 1e-5 ; 
    assert( std::abs( ppt.x() - pt.x ) < eps ); 
    assert( std::abs( ppt.y() - pt.y ) < eps ); 
    assert( std::abs( ppt.z() - pt.z ) < eps ); 

    // convert back from G4 -> glm 

    glm::mat4 trs2 = X4Transform3D::Convert(t0) ; 

    // compare the glm matrices componentwise

    float diff = nglmext::compDiff(trs, trs2); 
    std::cout 
           << gpresent("trs2", trs2) 
           << std::endl
           << " diff " << diff
           << std::endl
           ;

    assert( diff < eps ); 

}


void test_transform_0(float ax, float ay, float az, float angle_, float tx, float ty, float tz)
{
    glm::vec4 axis_angle(ax,ay,az, angle_ * CLHEP::pi/180.f );
    glm::vec3 tlat(tx,ty,tz) ; 
    glm::vec3 scal(1,1,1) ; 
    std::string order = "trs" ; 

    glm::mat4 mat(1.f) ;
    mat = glm::translate(mat, tlat );
    mat = glm::rotate(mat, axis_angle.w, glm::vec3(axis_angle) );
    // this does the translate last, despite appearances

    LOG(info) << glm::to_string(mat) ; 

   // glm::mat4 mat(1.f) ;
   // for(unsigned i=0 ; i < order.length() ; i++)
   // {   
   //     switch(order[i])
   //     {  
   //         case 's': mat = glm::scale(mat, scal)         ; break ; 
   //         case 'r': mat = glm::rotate(mat, axis_angle.w , glm::vec3(axis_angle)) ; break ;
   //         case 't': mat = glm::translate(mat, tlat )    ; break ;
   //     }  
   // }

    std::cout
            << "test_transform_0 " << order
            << std::endl 
            << gpresent( "axis_angle", axis_angle )
            << gpresent( "tlat", tlat )
            << gpresent( "mat", mat )
            << std::endl 
            ;

}

void test_transform_2(float ax, float ay, float az, float angle_, float tx, float ty, float tz)
{
    glm::vec3 tlat(tx,ty,tz);
    glm::vec4 trot(ax,ay,az, angle_);
    glm::vec3 tsca(1,1,1); 
 
    glm::mat4 trs = nglmext::make_transform("trs", tlat, trot, tsca );
    // trs: translate done last (despite "t" code setup coming first)  

    std::cout
            << "test_transform_2" 
            << std::endl 
            << gpresent( "tlat", tlat )
            << gpresent( "trot", trot )
            << gpresent( "tsca", tsca )
            << gpresent( "trs", trs )
            << std::endl 
            ;
}

void test_transform()
{
    float ax = 0 ; 
    float ay = 0 ; 
    float az = 1 ;

    float angle = 45.f ; 

    float tx = 100 ; 
    float ty = 0 ; 
    float tz = 0 ;

    test_transform_0(ax, ay, az,  angle, tx, ty, tz);
    test_transform_2(ax, ay, az,  angle, tx, ty, tz); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    test_transform();
    test_convert();
 
    return 0 ; 
}
