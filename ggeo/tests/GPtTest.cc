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

// TEST=GPtTest om-t

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GPt.hh"
#include "GPts.hh"


void test_make()
{
    GPts* pts = GPts::Make() ; 

    pts->add( new GPt( 101, 10001, 42,  "red" ) );
    pts->add( new GPt( 202, 20002, 43, "green" ) );
    pts->add( new GPt( 303, 30003, 44, "blue" ) );

    pts->dump();  

    const char* dir = "$TMP/ggeo/GPtsTest" ; 
    pts->save(dir); 

    GPts* pts2 = GPts::Load(dir); 
    pts2->dump(); 
}





int main(int argc, char** argv)
{
    unsigned ridx = argc > 1 ? std::atoi(argv[1]) : 0 ; 

    OPTICKS_LOG(argc, argv);

    //test_make(); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* idpath = ok.getIdPath() ;
    assert(idpath);  

    std::string objpath = ok.getObjectPath("GPts", ridx ); 
    LOG(info) 
        << std::endl 
        << " idpath  " << idpath << std::endl 
        << " objpath " << objpath << std::endl 
        ; 

    GPts* pts = GPts::Load(objpath.c_str()); 
    pts->dump(); 

    return 0 ;
}

