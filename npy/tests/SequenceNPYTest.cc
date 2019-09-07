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

#include <cassert>
#include <cstdlib>


#include "NPY.hpp"
#include "Index.hpp"
#include "Types.hpp"

#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "SequenceNPY.hpp"


/*
#ifdef SLOW_CPU_INDEXING
    SequenceNPY seq(dpho);
    seq.setTypes(&types);
    seq.setRecs(&rec);
    seq.setSeqIdx(evt.getRecselData());
    seq.indexSequences(32);   // creates and populates the seqidx CPU side 

    Index* seqhis = seq.getSeqHis();
    Index* seqmat = seq.getSeqMat();
    //seqidx->save("seq%s", typ, tag);  

*/



int main(int , char** )
{
    const char* idpath = getenv("IDPATH");
    const char* tag = "1" ;

    NPY<float>* photons = NPY<float>::load("oxcerenkov", tag, "dayabay");
    NPY<short>* records = NPY<short>::load("rxcerenkov", tag, "dayabay");
    NPY<float>* dom = NPY<float>::load("domain","1", "dayabay");
    NPY<int>*   idom = NPY<int>::load("idomain","1", "dayabay");

    if(!records) return 0 ;
    unsigned int maxrec = idom->getValue(0,0,3) ; 
    assert(maxrec == 10);

    records->dump("records NPY::dump");
    photons->dump("photons NPY::dump");


    Types types ; 
    types.dumpFlags();

    types.readMaterials(idpath, "GMaterialLib");
    types.dumpMaterials();

    bool flat = true ;   
    // getting this wrong, leads to photons being treated like records... 
    //     r.shape (6128410, 2, 4)   should be handled with flat = true  

    RecordsNPY r(records, maxrec, flat);
    r.setTypes(&types);
    r.setDomains(dom);

    SequenceNPY s(photons);
    {
        unsigned int nrec = records->getShape(0) ;
        NPY<unsigned char>* seqidx = NPY<unsigned char>::make(nrec, 2) ; 
        seqidx->zero();
        s.setSeqIdx(seqidx);
    }

    s.setTypes(&types);
    s.setRecs(&r);

    s.countMaterials();
    s.dumpUniqueHistories();
    s.indexSequences(32);

    NPY<unsigned long long>* seqhisnpy = s.getSeqHisNpy();
    seqhisnpy->save("$TMP/SequenceNPYTest_SeqHis.npy");

    NPY<unsigned char>* seqidx = s.getSeqIdx();
    seqidx->setVerbose();
    seqidx->save("$TMP/SequenceNPYTest_SeqIdx.npy");

    return 0 ;
}

