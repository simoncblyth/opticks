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
#include <cmath>


void test_mm(float a, float b)
{
    printf("fmaxf(%f,%f)    = %f\n", a,b,fmaxf(a,b));
    printf("fminf(%f,%f)    = %f\n", a,b,fminf(a,b));
}

 
int main(void)
{

    float a = 1.f/0.f ; 
    float b = 1.f ; 

    test_mm( 1.f/0.f, 1.f );
    test_mm( 1.f    , 1.f/0.f );

    test_mm(  1.f/0.f, -1.f/0.f );
    test_mm( -1.f/0.f,  1.f/0.f );



    return 0 ; 
}
