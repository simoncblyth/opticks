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
#include "UseGLM.hh"

#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/component_wise.hpp>


void test_UseGLM_camera()
{
    float tr = 100.f ; 
    glm::vec2 rot(10,10) ; 
    glm::mat4 pvm = UseGLM::camera( tr, rot ); 
    std::cout << glm::to_string(pvm) << std::endl ; 
}



void test_rotate_0()
{    
    /**
                 z
                 |
                 |
                 |
                 +----- y 
                /  
               /
              x

    **/ 

    glm::vec3 a(0.f, 0.f, 1.f);  // Z 
    glm::vec3 b(1.f, 0.f, 0.f);  // X

    glm::vec3 axb = glm::cross(a, b);   // cross product of Z ^ X = Y

    float la = glm::length(a) ;     // 1.
    float lb = glm::length(b) ;     // 1. 
    float ab = glm::dot(a, b);      // 0. (Z and X are perpendicualar) 
    float cos_angle = ab/(lb*la) ;  // 0. 
    float angle_radians = acos(cos_angle);  //  pi/2 radians

    std::cout << "         a: " << glm::to_string(a) << " glm::length(a) " << la << std::endl ;  
    std::cout << "         b: " << glm::to_string(b) << " glm::length(b) " << lb << std::endl ;  
  
    std::cout << "        ab: " << ab << " glm::dot(a, b) " << std::endl ; 
    std::cout << " cos_angle: " << cos_angle << std::endl ; 
    std::cout << "     angle: " << angle_radians << "(radians)" << std::endl ; 

    std::cout << "       axb: " << glm::to_string(axb) << " glm::cross(a, b) " << std::endl ;  

    glm::mat4 rot = glm::rotate(angle_radians, axb);  // matrix to rotate a->b from angle_radians and axis to rotate around (Y)   
    std::cout << "rot:" << glm::to_string(rot) << std::endl ;  

    glm::vec4 av(a, 0.f); 
   
    glm::vec4 bv0 = av * rot ; 
    glm::vec4 bv1 = rot * av  ; 
    std::cout << "        av: " << glm::to_string(av) << std::endl ;  
    std::cout << "        bv0: " << glm::to_string(bv0) << std::endl ;  
    std::cout << "        bv1: " << glm::to_string(bv1) << std::endl ;  
}


// promoted this to nglmext::make_rotate_a2b
glm::mat4 make_rotate_a2b(const glm::vec3& a, const glm::vec3& b)
{
    float ab = glm::dot(a, b);  // 0. when perpendicualar
    float la = glm::length(a) ;    
    float lb = glm::length(b) ;      
    float cos_angle = ab/(lb*la) ;  // 0. when perpendicular,  1. or -1. when collinear
    float angle_radians = acos(cos_angle);  //  pi/2 when perpendicular

    glm::vec3 axb = glm::cross(a, b);   
    // perpendicular to both a and b : axis around which to rotate a to b 

    glm::mat4 rot = glm::rotate(angle_radians, axb);  
    // matrix to rotate a->b from angle_radians and axis 

    return rot ; 
}

void test_make_rotate_a2b_2()
{
    std::cout << "test_make_rotate_a2b_2 (what happens when a and b are collinear)" << std::endl ; 

    glm::vec3 a(0.f, 0.f,   1.f);  // Z 
    glm::vec3 b(0.f, 0.f,  -1.f);  // -Z  

    glm::vec3 axb = glm::cross(a, b);   
    std::cout << "axb:" << glm::to_string(axb) << std::endl ;  

    float len_axb = glm::length(axb); 
    std::cout << "len_axb:" << len_axb << std::endl ;  

    glm::mat4 rot = make_rotate_a2b(a, b); 

    std::cout << "  a:" << glm::to_string(a) << std::endl ;  
    std::cout << "  b:" << glm::to_string(b) << std::endl ;  
    std::cout << "rot:" << glm::to_string(rot) << std::endl ;  
}

void test_make_rotate_a2b_1()
{
    std::cout << "test_make_rotate_a2b_1" << std::endl ; 

    glm::vec3 a(0.f, 0.f, 1.f);  // Z 
    glm::vec3 b(1.f, 0.f, 0.f);  // X

    glm::mat4 rot = make_rotate_a2b(a, b); 

    std::cout << "  a:" << glm::to_string(a) << std::endl ;  
    std::cout << "  b:" << glm::to_string(b) << std::endl ;  
    std::cout << "rot:" << glm::to_string(rot) << std::endl ;  

    glm::vec4 av(a, 0.f);  // w=0.f for direction 
    glm::vec4 rot_av = rot * av ; 
    glm::vec4 av_rot = av * rot ; 

    std::cout << " rot*av:" << glm::to_string(rot_av) << " (this way yields expected b vector) " << std::endl ;  
    std::cout << " av*rot:" << glm::to_string(av_rot) << " (nope)" << std::endl ;  
}


// promoted this to nglmext::make_rotate_a2b_then_translate
glm::mat4 make_rotate_a2b_then_translate( const glm::vec3& a, const glm::vec3& b, const glm::vec3& tlat )
{
    glm::mat4 rotate= make_rotate_a2b(a,b);     
    glm::mat4 translate = glm::translate(glm::mat4(1.0f), tlat );
    glm::mat4 rotate_then_translate = translate * rotate ; 
    return rotate_then_translate  ; 
}

void test_make_rotate_a2b_then_translate_0()
{
    std::cout << "test_make_rotate_a2b_then_translate_0" << std::endl ; 
    float epsilon = 1e-6 ; 

    glm::vec3 a(0.f, 0.f, 1.f);  // Z 
    glm::vec3 b(1.f, 0.f, 0.f);  // X
    glm::vec3 tlate(1000.f, 1000.f, 2000.f); 

    glm::mat4 tr = make_rotate_a2b_then_translate(a,b,tlate); 
    std::cout << "  tr:" << glm::to_string(tr) << std::endl ;  

    {
        // apply the matrix to the origin and check get expected result 

        glm::vec4 o(0.f,0.f,0.f,1.f);   

        glm::vec4 tr_o = tr * o ; 
        glm::vec4 tr_o_expected = glm::vec4(tlate, 1.f); 

        glm::vec4 o_tr = o * tr ; 

        std::cout << "           o_tr :" << glm::to_string(o_tr) << " (wrong order)" << std::endl ;  
        std::cout << "           tr_o :" << glm::to_string(tr_o) << std::endl ;  
        std::cout << "  tr_o_expected :" << glm::to_string(tr_o_expected) << std::endl ;  

        float o_diff = glm::compMax(glm::abs(tr_o - tr_o_expected)) ; 
        std::cout << " o_diff " << o_diff << std::endl ; 
        assert( o_diff < epsilon );
    }

    {
        // apply the matrix to a point along +Z and check get expected result 
        // rotation puts it along +X and then the translation is done 

        glm::vec4 q(0.f, 0.f, 100.f, 1.f); 
        glm::vec4 tr_q = tr * q ; 
        glm::vec4 tr_q_expected = glm::vec4(tlate, 1.f) + glm::vec4(100.f, 0.f, 0.f, 0.f ) ;  

        std::cout << "           tr_q :" << glm::to_string(tr_q) << std::endl ;  
        std::cout << "  tr_q_expected :" << glm::to_string(tr_q_expected) << std::endl ;  

        float q_diff = glm::compMax(glm::abs(tr_q - tr_q_expected)) ; 
        std::cout << " q_diff " << q_diff << std::endl ; 
        assert( q_diff < epsilon );
    }
}



int main(int argc, char** argv)
{
    //test_UseGLM_camera();
    //test_rotate_0(); 
    //test_make_rotate_a2b_1(); 
    test_make_rotate_a2b_2(); 
    //test_make_rotate_a2b_then_translate_0(); 

    return 0 ; 
}
