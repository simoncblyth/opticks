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

// op --gitemindex 

#include <string>
#include <iostream>
#include <cassert>

// brap-
#include "BOpticksEvent.hh"

// npy-
#include "Types.hpp"
#include "Index.hpp"

// okc-
#include "OpticksEvent.hh"
#include "OpticksConst.hh"

// ggeo-
#include "GItemIndex.hh"

#include "OPTICKS_LOG.hh"



void dump(GItemIndex* idx, const char* msg)
{
    assert(idx);
    std::cout << idx->gui_radio_select_debug();
    idx->test();
    idx->dump(msg);
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    

    Types types ; 

    const char* pfx = "source" ; 
    const char* det = "concentric" ; 
    const char* typ = "torch" ; 
    const char* tag = "1" ;

    std::string tagdir_ = OpticksEvent::TagDir( pfx, det, typ, tag );  
    const char* tagdir = tagdir_.c_str() ; 

    LOG(info) << argv[0] << " tagdir " << tagdir ;  


    const char* reldir = NULL ; 


    if(1)
    {
        Index* seqhis = Index::load(tagdir, OpticksConst::SEQHIS_NAME_, reldir);
        if(!seqhis)
        {
            LOG(error) << " NULL seqhis " ; 
            return 0 ;  
        } 

        GItemIndex* m_seqhis = new GItemIndex(seqhis);
        m_seqhis->setTypes(&types);
        m_seqhis->setLabeller(GItemIndex::HISTORYSEQ);
        m_seqhis->formTable();
        dump(m_seqhis, "m_seqhis");


        int* ptr = seqhis->getSelectedPtr();

        for(unsigned i=0 ; i < seqhis->getNumKeys() ; i++)
        {
            *ptr = i ; 
            const char* key = m_seqhis->getSelectedKey();
            const char* label = m_seqhis->getSelectedLabel();
            LOG(info) << " i " << std::setw(5) << i 
                      << " key: " << std::setw(30) << key 
                      << " label: " << std::setw(60) << label
                      ; 
        }


    }
    if(0)
    {
        Index* seqmat = Index::load(tagdir, OpticksConst::SEQMAT_NAME_, reldir );
        GItemIndex* m_seqmat = new GItemIndex(seqmat);
        m_seqmat->setTypes(&types);
        m_seqmat->setLabeller(GItemIndex::MATERIALSEQ);
        m_seqmat->formTable();
        dump(m_seqmat, "m_seqmat");
    }
    if(0)
    {
        Index* bndidx = Index::load(tagdir, OpticksConst::BNDIDX_NAME_, reldir );
        GItemIndex* m_bndidx = new GItemIndex(bndidx);
        dump(m_bndidx,"m_bndidx");
    }


    return 0 ;
}
