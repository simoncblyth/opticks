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

#include <cstdio>
#include <cassert>
#include <vector>



#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"

#include "PLOG.hh"


void test_gmat4()
{
    std::string s = "0.500,-0.866,0.000,-86.603,0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000," ;
    glm::mat4 m = gmat4(s);
    print(m, "mat4");
}

void test_ivec4()
{
    std::vector<std::string> ss ; 

    //ss.push_back("");     // bad-lexical case TODO: trap this
    ss.push_back("1");    
    ss.push_back("1,2");    
    ss.push_back("1,2,3");    
    ss.push_back("1,2,3,4");    
    ss.push_back("1,2,3,4,5");    

    for(unsigned int i=0 ; i < ss.size() ; i++)
    {
         std::string s = ss[i];
         glm::ivec4 v = givec4(s);
         print(v, s.c_str());
    }
}

void test_misc()
{
    std::string sv = "1,2,3" ;
    glm::vec3 v = gvec3(sv);     
    print(v, "gvec3(sv)");

    std::string sq = "1,2,3,4" ;
    glm::quat q = gquat(sq);     
    print(q, "gquat(sq)");

    std::string sqq = gformat(q);
    printf("%s\n", sqq.c_str());

    glm::quat qq = gquat(sqq);
    print(qq, "qq:gquat(sqq)");

    assert( q == qq );
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_gmat4();
    test_ivec4();
    test_misc();

    return 0 ; 
}

