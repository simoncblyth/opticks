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

#include "SPath.hh"
#include "SSys.hh"
#include "SSys.hh"

#include "LaunchCommon.hh"
#include "LaunchSequence.hh"
#include "cuRANDWrapper.hh"

#include "OPTICKS_LOG.hh"


//#define WORK 1024*768
#define WORK 1024*1

/*
TODO: move to BOpticksResource, not envvars 

Improving this is stymied, as need access to BOpticksResource 
to avoid envvar crutch, but have prejudice against use of boost 
in cudarap- (is that prejudice still justified, now that I know 
better how to split things between host compiler and nvcc ?).

*/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned int work              = SSys::getenvint("CUDARAP_RNG_MAX", WORK) ;
    unsigned long long seed        = 0 ;
    unsigned long long offset      = 0 ;
    unsigned int max_blocks        = SSys::getenvint("MAX_BLOCKS", 128) ;
    unsigned int threads_per_block = SSys::getenvint("THREADS_PER_BLOCK", 256) ; 

    int create_dirs = 2 ; // 2:directory path argument
    const char* tmp = SPath::Resolve("$TMP", create_dirs ); 
    const char* cachedir = SSys::getenvvar("CUDARAP_RNG_DIR", tmp) ;

    LOG(info) 
          << " work " << work 
          << " max_blocks " << max_blocks
          << " seed " << seed 
          << " offset " << offset
          << " threads_per_block " << threads_per_block
          << " cachedir " << cachedir 
          ;


    cuRANDWrapper* crw = cuRANDWrapper::instanciate( work, cachedir, seed, offset, max_blocks, threads_per_block );

    crw->Allocate();
    crw->InitFromCacheIfPossible(); 
    // CAUTION: without Init still provides random numbers but different ones every time

    // can increase max_blocks as generation much faster than initialization 

    const LaunchSequence* launchseq = crw->getLaunchSequence() ;
    const_cast<LaunchSequence*>(launchseq)->setMaxBlocks(max_blocks*32);  

    crw->Test();

    crw->Summary("cuRANDWrapperTest::main");


    crw->resize( 1024*10 );

    crw->Test();

    crw->Summary("cuRANDWrapperTest::main after resize");


    return 0 ; 
}

