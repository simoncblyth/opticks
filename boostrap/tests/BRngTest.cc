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
#include "BRng.hh"
#include "OPTICKS_LOG.hh"


void test_rand()
{
    std::cout << " RAND_MAX " << RAND_MAX << std::endl ; 

    for(int i=0 ; i < 10 ; i++ )
    std::cout << rand() << std::endl ; 

}


void test_separate()
{

    BRng a(0,1, 42, "A") ; 
    a.dump();

    BRng b(0,1, 42, "B") ; 
    b.dump();


/*
    BRng c(0,100, 42, "C") ;
    c.dump();

    BRng d(0,100, 42, "D") ;
    d.dump();

    BRng e(0,1, 42, "E") ;
    e.dump();

    BRng f(0.9,1, 42, "F") ;
    f.dump();
*/
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_rand();
    //test_separate();


    return 0 ; 
}


