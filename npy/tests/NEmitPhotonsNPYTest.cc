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

/*
NEmitPhotonsNPYTest $TMP/tboolean-torus--

NEmitPhotonsNPYTest $TMP/tboolean-box--

NEmitPhotonsNPYTest $TMP/tboolean-box-- 1,5,99992

*/

#include <cstdlib>
#include <cfloat>

#include "SSys.hh"
#include "NGLMExt.hpp"

#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "BRAP_LOG.hh"
#include "SYSRAP_LOG.hh"

#include "NPY.hpp"
#include "NCSGList.hpp"
#include "NEmitPhotonsNPY.hpp"
#include "NPho.hpp"
#include "NPY.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    SYSRAP_LOG__ ;  
    NPY_LOG__ ;  
    BRAP_LOG__ ;  

    const char* csgpath = argc > 1 ? argv[1] : NULL ; 
    const char* maskstr = argc > 2 ? argv[2] : NULL ; 

    if(csgpath == NULL)
    {
        LOG(warning) << "Expecting 1st argument csgpath directory containing NCSG trees" ; 
        return 0 ;
    } 

    unsigned verbosity = 0 ; 
    NCSGList* trees = NCSGList::Load(csgpath, verbosity );    

    NCSG* csg = trees->findEmitter();

    unsigned numTrees = trees->getNumTrees() ;
    LOG(info) << " numTrees " << numTrees 
              << " emitter " << csg
              ; 

    if(csg == NULL)
    {
        LOG(warning) << " failed to find emitter in those trees " ; 
        return 0 ; 
    }

    unsigned EMITSOURCE = 0x1 << 18 ; 
    unsigned gencode = EMITSOURCE ; 
    unsigned seed = 42 ; 
    bool emitdbg = false ; 

    NPY<unsigned>* mask = maskstr ? NPY<unsigned>::make_from_str(maskstr) : NULL  ; 
    if(mask) mask->dump("mask") ; 


    NEmitPhotonsNPY ep(csg, gencode, seed, emitdbg, mask) ;


    NPY<float>* ox = ep.getPhotonsRaw();
    ox->dump("ox-raw");

    NPY<float>* gs = ep.getFabStepRawData();
    gs->dump("fs-raw");

        
    NPY<float>* oxm = ep.getPhotons() ;
    oxm->dump("ox-maybe-masked");

    NPY<float>* gsm = ep.getFabStepData();
    gsm->dump("fs-maybe-masked");



    const char* path = "$TMP/NEmitPhotonsNPYTest_fabstep.npy" ;
    gs->save(path);
    SSys::npdump(path, "np.int32");

    const char* path_masked = "$TMP/NEmitPhotonsNPYTest_fabstep_masked.npy" ;
    gsm->save(path_masked);
    SSys::npdump(path_masked, "np.int32");


    NPho::Dump(ox, 10000, 10, "ox" ) ; // modulo, margin

    NPho::Dump(oxm, 10000, 10, "oxm" ) ; // modulo, margin


    return 0 ; 
}


