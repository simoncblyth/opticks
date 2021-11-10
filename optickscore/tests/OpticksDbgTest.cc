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


// OpticksDbgTest --OKCORE trace

#include <cassert>
#include <cstdlib>

#include "NPY.hpp"
#include "Opticks.hh"
#include "OPTICKS_LOG.hh"


void test_isDbgPhoton_string(int argc, char** argv)
{
    Opticks ok(argc, argv, "--dindex 1,10,100,200");
    ok.configure();

    assert(ok.isDbgPhoton(1) == true );
    assert(ok.isDbgPhoton(10) == true );
    assert(ok.isDbgPhoton(100) == true );
    assert(ok.isDbgPhoton(200) == true );

    const std::vector<unsigned>& dindex = ok.getDbgIndex();

    assert(dindex.size() == 4);
    assert(dindex[0] == 1);
    assert(dindex[1] == 10);
    assert(dindex[2] == 100);
    assert(dindex[3] == 200);
}

void test_isDbgPhoton_path(int argc, char** argv)
{
    Opticks ok(argc, argv, "--dindex $TMP/c.npy");
    ok.configure();
    if(ok.getNumDbgPhoton() == 0 ) return ; 

    assert(ok.isDbgPhoton(268) == true );
    assert(ok.isDbgPhoton(267) == false );
}

void test_getMaskBuffer(int argc, char** argv)
{
    Opticks ok(argc, argv, "--mask 1,3,5,7,9");
    ok.configure();

    NPY<unsigned>* msk = ok.getMaskBuffer() ;

    assert( msk && msk->getShape(0) == 5 );
    msk->dump("msk");
}

void test_postconfigure_strings(int argc, char** argv)
{
    unsetenv("OPTICKS_KEY"); 
    Opticks ok(argc, argv, "--allownokey --x4skipsolidname one,two,three") ; 
    ok.configure();

    assert( ok.isX4SkipSolidName("one") == true ); 
    assert( ok.isX4SkipSolidName("two") == true ); 
    assert( ok.isX4SkipSolidName("three") == true ); 
    assert( ok.isX4SkipSolidName("four") == false ); 

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_isDbgPhoton_string(argc, argv);
    //test_isDbgPhoton_path(argc, argv);
    //test_getMaskBuffer(argc, argv);

    test_postconfigure_strings(argc, argv) ; 

    return 0 ; 
}
