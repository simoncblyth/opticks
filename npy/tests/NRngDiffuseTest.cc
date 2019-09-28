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

#include "NPY.hpp"
#include "GLMFormat.hpp"
#include "NRngDiffuse.hpp"
#include "OPTICKS_LOG.hh"


const char* TMPDIR = "$TMP/npy/NRngDiffuseTest" ; 

void test_uniform_sphere(NRngDiffuse& rng, const glm::vec3& dir)
{
    LOG(info) << "." ; 
    glm::vec4 u ; 
    for(unsigned i=0 ; i < 10 ; i++)
    {
        rng.uniform_sphere(u);
        float udotd = glm::dot(glm::vec3(u), dir); 
        std::cout << gpresent(u) << " " << udotd << std::endl ; 
    } 
}

void test_diffuse(NRngDiffuse& rng, const glm::vec3& dir)
{
    LOG(info) << "." ; 
    glm::vec4 u ; 
    int trials ; 
    for(unsigned i=0 ; i < 10 ; i++)
    {
        float udotd = rng.diffuse(u, trials, dir);
        std::cout << gpresent(u) << " " << udotd << " " << trials << std::endl ; 
    } 
}


void test_uniform_sphere_sample(NRngDiffuse& rng)
{
    NPY<float>* samp = rng.uniform_sphere_sample(1000) ;
    samp->dump();
    samp->save(TMPDIR, "NRngDiffuseDiffuseTest_sphere.npy");
}

void test_diffuse_sample(NRngDiffuse& rng, const glm::vec3& dir)
{
    NPY<float>* samp = rng.diffuse_sample(1000, dir) ;
    samp->dump();
   
    samp->save(TMPDIR, "NRngDiffuseDiffuseTest_diffuse.npy");
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    glm::vec3 dir(0,1,0); 
    dir = glm::normalize(dir );    
    std::cout << "dir:" << gpresent(dir) << std::endl ; 

    unsigned seed = 42 ; 

    float ctmin = 0.f ; 
    float ctmax = 1.f ; 

    NRngDiffuse rng(seed, ctmin, ctmax) ; 
    LOG(info) << rng.desc() ; 

    test_uniform_sphere(rng, dir) ; 
    test_diffuse(rng, dir) ; 

    test_uniform_sphere_sample(rng) ; 
    test_diffuse_sample(rng, dir) ; 
  

    return 0 ; 
}

/*

simon:opticksnpy blyth$ NRngDiffuseTest 
dir:(      0.000     1.000     0.000)
2017-12-01 16:46:32.099 INFO  [905346] [main@52] NRngDiffuse seed 42 uct.lo 0.5 uct.hi 1
2017-12-01 16:46:32.099 INFO  [905346] [test_uniform_sphere@7] .
(     -0.450     0.453    -0.770) 0.45256
(      0.288    -0.958    -0.006) -0.957525
(      0.929    -0.297     0.218) -0.297407
(      0.247     0.555    -0.794) 0.555314
(     -0.077    -0.676    -0.733) -0.675646
(      0.132    -0.699    -0.702) -0.69938
(     -0.696    -0.497    -0.519) -0.49664
(     -0.619    -0.432    -0.656) -0.431598
(      0.522     0.779    -0.346) 0.77944
(     -0.738     0.261     0.622) 0.26136
2017-12-01 16:46:32.100 INFO  [905346] [test_diffuse@19] .
(     -0.489     0.843     0.224) 0.842815 6
(      0.612     0.768     0.190) 0.767679 2
(      0.230     0.853    -0.467) 0.853497 1
(      0.506     0.704     0.499) 0.704032 1
(     -0.431     0.757     0.491) 0.756795 5
(     -0.328     0.925    -0.190) 0.925311 8
(     -0.186     0.703     0.686) 0.702914 6
(      0.107     0.988     0.110) 0.988099 9
(     -0.206     0.918    -0.339) 0.917744 3
(      0.036     0.645    -0.763) 0.645258 27


*/


