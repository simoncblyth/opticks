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
#include <fstream>
#include <sstream>

#include "CVec.hh"
#include "G4MaterialPropertyVector.hh"
#include "OPTICKS_LOG.hh"

#include "SDirect.hh"

// this was formerly named G4PhysicsOrderedFreeVectorTest.cc


void test_redirected( G4MaterialPropertyVector& vec, bool ascii )
{
    std::ofstream fp("/dev/null", std::ios::out); 
    std::stringstream ss ;     
    stream_redirect rdir(ss,fp); // stream_redirect such that writes to the file instead go to the stringstream 
    
    vec.Store(fp, ascii );

    std::cout <<  ss.str() << std::endl ; 
}



void test_caveman( G4MaterialPropertyVector& vec, bool ascii )
{
    std::vector<char> buf(512);
    for(unsigned j=0 ; j < buf.size() ; j++ ) buf[j] = '*' ; 

    std::ofstream fp("/dev/null", std::ios::out); 

    fp.rdbuf()->pubsetbuf(buf.data(),buf.size());

    vec.Store(fp, ascii );

    for(unsigned j=0 ; j < buf.size() ; j++ )
    {
        std::cout << " " << buf[j] ; 
        if( (j + 1) % 16 == 0 ) std::cout << std::endl ; 
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    CVec* v = CVec::MakeDummy(5); 

    G4MaterialPropertyVector& vec  = *v->getVec() ; 

    std::cout << vec << std::endl ; 


    // Making an ofstream writing method write into a buffer 

    bool ascii = false ; 
    //test_caveman(   v, ascii ); 
    test_redirected( vec, ascii ); 



    return 0 ; 
}


