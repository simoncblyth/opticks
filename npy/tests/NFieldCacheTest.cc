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

#include "NGenerator.hpp"
#include "NFieldCache.hpp"
#include "NPY_LOG.hh"
#include "NBox.hpp"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    nbox* world = nbox::Create(0,0,0,100, CSG_BOX) ; 
    nbbox wbb = world->bbox() ;
    NGenerator gen(wbb);
    nbox* obj = nbox::Create(0,0,0,10, CSG_BOX) ; 

    NFieldCache fc(*obj, wbb) ; 

    std::function<float(float,float,float)> fn = fc.func();


    nvec3 p ; 
    for(int i=0 ; i < 1000 ; i++ )
    {
        gen(p);

        for(int j=0 ; j < 10 ; j++)
        {
            float v0 = (*obj)(p.x, p.y, p.z) ;
            float v1 = fn(p.x, p.y, p.z) ;

            if(i % 100 == 0)
            {
              LOG(info) 
                 << " i " << std::setw(6) << i 
                 << " p " << p.desc()
                 << " obj: " << v0 
                 << " fc: "  << v1
              ;
            } 
        }
    }

    LOG(info) << fc.desc() ;  
    return 0 ; 
}
