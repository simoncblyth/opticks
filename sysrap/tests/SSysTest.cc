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

#include <sstream>
#include <string>
#include "SSys.hh"

#include "OPTICKS_LOG.hh"


int test_tpmt()
{
    return SSys::run("tpmt.py");
}

int test_RC(int irc)
{
    //assert( irc < 0xff && irc >= 0 ) ; 
    std::stringstream ss ; 
    ss << "python -c 'import sys ; sys.exit(" << irc << ")'" ;
    std::string s = ss.str();
    return SSys::run(s.c_str());
}


void test_RC()
{
    int rc(0);
    for(int irc=0 ; irc < 500 ; irc+=10 )
    {
        int xrc = irc & 0xff ;   // beyond 0xff return codes get truncated 
        rc = test_RC(irc);       
        assert( rc == xrc ); 
    } 
}

int test_OKConfCheck()
{
    int rc = SSys::OKConfCheck();
    assert( rc == 0 );
    return rc ; 
}

void test_DumpEnv()
{
    SSys::DumpEnv("OPTICKS"); 
}

void test_IsNegativeZero()
{
    float f = -0.f ; 
    float z = 0.f ; 

    assert( SSys::IsNegativeZero(f) == true ); 
    assert( SSys::IsNegativeZero(z) == false ); 
    assert( SSys::IsNegativeZero(-1.f) == false ); 
}


int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    int rc(0) ;

    //rc = test_OKConfCheck();

    //rc = test_tpmt();

    //rc = test_RC(77);

    //LOG(info) << argv[0] << " rc " << rc ; 
   
    //test_DumpEnv();

    test_IsNegativeZero(); 

    return rc  ; 
}

