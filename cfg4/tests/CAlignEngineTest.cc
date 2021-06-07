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

// TEST=CAlignEngineTest om-t 

#include <iostream>
#include <iomanip>

#include "OPTICKS_LOG.hh"
#include "CAlignEngine.hh"
#include "NPY.hpp"
#include "Randomize.hh"

/**
CAlignEngineTest
==================

* NB: the more dynamic TCURAND based CRandomEngine is in the process of replacing CAlignEngine

CAlignEngine loads from $TMP/TRngBufTest_0.npy requiring 
TRngBufTest to have been run previously.

The buffer sizes at creation need to match those
asserted on below, see thrustrap/tests/TRngBufTest.cu


**/


struct CAlignEngineTest
{
    CAlignEngineTest();  
    void burn(unsigned n, bool dump);
    void check( int v0 , int v1, int i0, int i1, bool dump );
    void brief(int v0, int v1);


    bool ready ; 
    const CAlignEngine* ae ;  
    NPY<double>* seq ; 
    int ni ; 
    int nv ; 

};


CAlignEngineTest::CAlignEngineTest()
    :
    ready(CAlignEngine::Initialize("$TMP/cfg4/CAlignEngineTest")),  // logfile is written to this dir
    ae(CAlignEngine::INSTANCE),
    seq(ae->m_seq),        // pre-cooked randoms
    ni(ae->m_seq_ni),      // number of record lines 
    nv(ae->m_seq_nv)       // number of values for each record line
{
    LOG(info) 
        << " ni " << ni 
        << " nv " << nv
        ;

    assert( seq ); 
    assert( nv == 256 );      
    assert( ni == 10000 );      
}

void CAlignEngineTest::brief(int v0, int v1)
{
    LOG(info) << " v0 " << v0 << " v1 " << v1 ; 
    burn(10, true) ; 
    check( v0, v1, 0, 10, true ); 
    burn(10, true) ; 
    check( v0, v1, ni-10, ni, true ); 
}




/**
CAlignEngineTest::burn
-------------------------

Calls G4UniformRand n times

**/

void CAlignEngineTest::burn(unsigned n, bool dump)
{
    for(unsigned i=0 ; i < n ; i++)
    {
        double u = G4UniformRand() ; 
        if(dump) 
           std::cout 
               << " " << std::setw(6) << i 
               << ":" 
               << " " << std::setw(10) << std::fixed << std::setprecision(6) << u  
               << std::endl 
               ;

    }
}


/**
CAlignEngineTest::check
-------------------------

For each value in the sequence v0:v1 hop between stream lines i0:i1, 
ie the outer loop is over positions into the sequence and
the inner one is over sequence lines (as used by different
photon record_id). 

This works by calling CAlignEngine::SetSequenceIndex(i) 
to hop between streams and checks the results from G4UniformRand() 
matches against expectations from the pre-cooked buffer of randoms.


**/

void CAlignEngineTest::check( int v0 , int v1, int i0, int i1, bool dump )
{

    //assert( v0 == 0 && " have to start from value zero for the check to match "); 

    if(dump)
    {
        std::cout 
            << " v0 " << v0 
            << " v1 " << v1
            << " i0 " << i0 
            << " i1 " << i1
            << std::endl 
            ;

        std::cout << std::setw(10) << "" << "   " ; 
        for(int i=i0 ; i < i1 ; i++) std::cout << " " <<  std::setw(10) << i ; 
        std::cout << std::endl ; 
    } 

    for(int v=v0 ; v < v1 ; v++)
    {
        if(dump)
            std::cout << std::setw(10) << v << " : " ; 


        for(int i=i0 ; i < i1 ; i++)
        {
            CAlignEngine::SetSequenceIndex(i); 

            double u = G4UniformRand() ; 

            if(dump) std::cout << " " << std::setw(10) << std::fixed << std::setprecision(6) << u  ;
            if( i >= 0 )
            {
                double u2 = seq->getValue(i, 0, 0, v);
                //if(dump) std::cout << " (" << std::setw(10) << std::fixed << std::setprecision(6) << u2 << ")" ; 

                bool match = u == u2 ; 
                if(!match)
                    LOG(fatal) 
                       << " mismatch "
                       << " v " << v
                       << " i " << i
                       << " u " << u 
                       << " u2 " << u2
                       ; 

                assert(match); 
            }

            CAlignEngine::SetSequenceIndex(-1);   
            //   essential otherwise subsequent burns will consume from the last set SequenceIndex that will 
            //   cause subsequent checks to mismatch 
        }
        if(dump) std::cout << std::endl  ;
    } 
}


int main( int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CAlignEngineTest aet ; 
    aet.brief(0, 16); 
    aet.brief(16,32); 


    return 0 ; 
}
