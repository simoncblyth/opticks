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

// TEST=G4UniformRandTest om-t

/*
   g4-;g4-cls Randomize
   g4-;g4-cls Random
   g4-;g4-cls RandomEngine
   g4-;g4-cls JamesRandom
   g4-;g4-cls MixMaxRng 


simon:Random blyth$ grep public\ HepRandomEngine *.*
DualRand.h:class DualRand: public HepRandomEngine {
JamesRandom.h:class HepJamesRandom: public HepRandomEngine {
MTwistEngine.h:class MTwistEngine : public HepRandomEngine {
MixMaxRng.h:class MixMaxRng: public HepRandomEngine {
NonRandomEngine.h:class NonRandomEngine : public HepRandomEngine {
RanecuEngine.h:class RanecuEngine : public HepRandomEngine {
Ranlux64Engine.h:class Ranlux64Engine : public HepRandomEngine {
RanluxEngine.h:class RanluxEngine : public HepRandomEngine {
RanshiEngine.h:class RanshiEngine: public HepRandomEngine {
simon:Random blyth$ 

https://arxiv.org/pdf/1307.5869.pdf

http://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-overview

*/




#include <iostream>
#include "OPTICKS_LOG.hh"

#include "BFile.hh"
#include "BStr.hh"


#include "Randomize.hh"
#include "CLHEP/Random/NonRandomEngine.h"
#include "CLHEP/Random/MixMaxRng.h"

#include "NPY.hpp"


void dump_flat(int n)
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;

    long seed = engine->getSeed() ; 
    LOG(info) 
        << " seed " << seed 
        << " name " << engine->name() 
        ; 

    for(int i=0 ; i < n ; i++)
    {
        double u = engine->flat() ;   // equivalent to the standardly used: G4UniformRand() 
        std::cout << u << std::endl ; 
    }
}

NPY<double>* make_flat_array(int n)
{
    NPY<double>* a = NPY<double>::make(n) ; 
    a->zero();
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;
    engine->flatArray(n, a->getValues());  
    return a ; 
}


/**
test_resumption
-----------------

Repeated running of this will build a sequence 
in 10 parts, by saving/restoring the engine state.


**/

void test_resumption(const char* dir, int n)
{  
    std::string path_ = BFile::FormPath(dir, "engine.conf") ; 
    const char* path = path_.c_str(); 

    bool exists = BFile::ExistsFile(path) ; 
    long seed = CLHEP::HepRandom::getTheSeed(); 
    LOG(info) 
         << " path " << path 
         << " exists " << exists 
         << " seed " << seed
         ; 

    if(exists)
    {
        LOG(info) << " restore state from " << path  ; 
        CLHEP::HepRandom::restoreEngineStatus(path); 
    }
    else
    { 
        NPY<double>* a = make_flat_array(n*10);
        a->dump("full", n*10); 
        a->save(dir, "full.npy") ;
        LOG(info) << " first call, saving full into " << dir  ; 
    
        CLHEP::HepRandom::setTheSeed(seed) ;   // return to starting point in sequence by setting seed 
    }

    const char* name = NULL ; 
    for( int i = 0 ; i < 11 ; i++ )
    {
        name = i == 10 ? NULL : BStr::concat<int>("part", i, ".npy") ; 
        if(name == NULL || !BFile::ExistsFile(dir, name) ) break ; 
        free((void*)name) ;  
    }

    if( name == NULL )
    {
        LOG(info) << " set is complete " ; 
        return ; 
    }

    NPY<double>* b = make_flat_array(n);
    b->dump(name, n); 
    b->save(dir, name); 
    LOG(info) << " saving " << name ; 
    CLHEP::HepRandom::saveEngineStatus(path); 

}



void test_NonRandomEngine()
{
    unsigned N = 10 ;    // needs to provide all that are consumed
    std::vector<double> seq ; 
    for(unsigned i=0 ; i < N ; i++ ) seq.push_back( double(i)/double(N) );  

    CLHEP::NonRandomEngine*   custom_engine = new CLHEP::NonRandomEngine();
    custom_engine->setRandomSequence( seq.data(), seq.size() ) ; 

    CLHEP::HepRandom::setTheEngine( custom_engine );  

    long custom_seed = 9876 ;  // <-- ignored for NonRandom
    CLHEP::HepRandom::setTheSeed( custom_seed );  

    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;

    assert( engine == custom_engine ) ; 

    dump_flat(N); 

}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    //test_resumption("$TMP/cfg4/G4UniformRandTest", 10 ); 



    //test_NonRandomEngine();
    //test_DefaultEngine();

/*
    //CLHEP::HepJamesRandom* custom_engine = new CLHEP::HepJamesRandom();
    //CLHEP::MTwistEngine*   custom_engine = new CLHEP::MTwistEngine();
*/

    return 0 ; 
}



