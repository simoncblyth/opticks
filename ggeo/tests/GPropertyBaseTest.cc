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

#include "GProperty.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    int N = 3 ; 

    float* val = new float[N] ;
    float* dom = new float[N] ;
  
    for(int i=0 ; i < N ; i++)
    {
        dom[i] = i*1.f ;   
        val[i] = i*10.f ;   
    }

    GProperty<float>* prop = new GProperty<float>(val,dom,N);
    prop->Summary();

    GProperty<float>* cprop = GProperty<float>::from_constant( 1.f , val, N );
    cprop->Summary(); 


    return 0 ; 
}
