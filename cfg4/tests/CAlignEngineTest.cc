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


struct CAlignEngineTest
{
    CAlignEngineTest();  
    void spin(unsigned n, bool dump);
    void check( int v0 , int v1, int i0, int i1, bool dump );


    bool ready ; 
    const CAlignEngine* ae ;  
    NPY<double>* seq ; 
    int ni ; 
    int nv ; 

};


CAlignEngineTest::CAlignEngineTest()
    :
    ready(CAlignEngine::Initialize("$TMP/cfg4/CAlignEngineTest")),
    ae(CAlignEngine::INSTANCE),
    seq(ae->m_seq),
    ni(ae->m_seq_ni), 
    nv(ae->m_seq_nv)  
{
    assert( seq ); 
    assert( nv == 256 );      
    assert( ni == 100000 );      

    LOG(info) 
        << " ni " << ni 
        << " nv " << nv
        ;

    spin(10, true) ; 

    check( 0, 16, 0, 10, true ); 
    //check(  0,  nv, 0, 10, true );     

    check( 0, 16, ni-10, ni, true ); 
}



void CAlignEngineTest::spin(unsigned n, bool dump)
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



void CAlignEngineTest::check( int v0 , int v1, int i0, int i1, bool dump )
{
    // for each value in the sequence hop between streams

    assert( v0 == 0 && " have to start from value zero for the check to match "); 

    // are not accessing something with an index, 
    // are making a sequence of calls to G4UniformRand()

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
        }
        if(dump) std::cout << std::endl  ;
    } 
}


int main( int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CAlignEngineTest aet ; 

    return 0 ; 
}
