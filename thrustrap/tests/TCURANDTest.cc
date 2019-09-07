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

// TEST=TCURANDTest om-t 

#include "SSys.hh"
#include "SStr.hh"
#include "NPY.hpp"
#include "BFile.hh"
#include "Opticks.hh"

#include "TCURAND.hh"
#include "OPTICKS_LOG.hh"


struct TCURANDTest
{
    unsigned          m_ni ; 
    TCURAND<double>*  m_tc ; 
    NPY<double>*      m_ox ;  

    TCURANDTest(unsigned ni)
        :
        m_ni( ni ), 
        m_tc( new TCURAND<double>( m_ni, 16, 16 )),
        m_ox( m_tc->getArray() )
    {
    }
    void test_one(unsigned ibase)
    { 
        m_tc->setIBase(ibase);  // invokes generate, updating the array
        save(); 
    }
    void test_many()
    {
        for(unsigned ibase=0 ; ibase <= m_ni*2 ; ibase += m_ni )
        {
            test_one(ibase); 
        }
    }
    const char* getPath()
    {
        unsigned ibase = m_tc->getIBase(); 
        const char* path = SStr::Concat("$TMP/TCURANDTest_", ibase, ".npy") ; 
        std::string spath = BFile::FormPath(path); 
        return strdup(spath.c_str()) ; 
    }
    void save()
    {
        const char* path = getPath();  
        LOG(info) << " save " << path ; 
        m_ox->save(path)  ;
        //SSys::npdump(path, "np.float64", NULL, "suppress=True,precision=8" );
    } 
};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    Opticks ok(argc, argv); 
    ok.configure(); 

    int IBASE = SSys::getenvint("IBASE", -1) ; 

    unsigned ni = 100*1000 ; 
    //unsigned ni = 1000 ; 

    TCURANDTest tct(ni) ;  

    if( IBASE < 0 ) 
    {
        tct.test_many(); 
    }
    else
    {
        tct.test_one(IBASE); 
    }


    ok.dumpProfile("TCURANDTest"); 

    return 0 ; 
}


