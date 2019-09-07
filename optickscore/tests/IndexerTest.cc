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

// op --tindexer

/**
IndexerTest
=============

::

   ckm-indexer-test 


**/

#include "NGLM.hpp"
#include "Opticks.hh"
#include "OpticksConst.hh"
#include "OpticksEvent.hh"
#include "Indexer.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* pfx = "source" ;  // 
    const char* typ = "natural" ; 
    const char* tag = "1" ; 
    const char* det = "g4live" ; 
    const char* cat = NULL ; 
 
    OpticksEvent* evt = OpticksEvent::load(pfx, typ, tag, det, cat) ;

    if(!evt) 
    {
        LOG(info) << " failed to load " ; 
        return 0 ; 
    }

    LOG(info) << evt->getShapeString() ; 

    NPY<unsigned long long>* sequence = evt->getSequenceData();
    NPY<unsigned char>*        phosel = evt->getPhoselData();
    assert(sequence->getShape(0) == phosel->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    idx->indexSequence(OpticksConst::SEQHIS_NAME_, OpticksConst::SEQMAT_NAME_);
    idx->applyLookup<unsigned char>(phosel->getValues());

    phosel->save("$TMP/phosel.npy"); 


    return 0 ; 
}
