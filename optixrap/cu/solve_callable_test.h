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

__device__ void solve_callable_test()
{
    double a(10) ;
    double b(20) ;
    double c(30) ;
    double xx[3] ; 

    unsigned msk = 0u ; 
    unsigned nr = solve_callable[0](a,b,c,xx,msk);

    rtPrintf("solve_callable_test:solve_callable[0] abc (%g %g %g) nr %u xx (%g %g %g) \n",
             a,b,c,nr,xx[0],xx[1],xx[2]
            );
 

} 
