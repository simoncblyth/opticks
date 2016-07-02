// op --gitemindex 

#include <string>
#include <iostream>
#include <cassert>

// npy-
#include "Types.hpp"
#include "Index.hpp"
#include "NPYBase.hpp"

// ggeo-
#include "GItemIndex.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"



void dump(GItemIndex* idx, const char* msg)
{
    assert(idx);
    std::cout << idx->gui_radio_select_debug();
    idx->test();
    idx->dump(msg);
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;
    

    Types types ; 

    const char* m_typ = "torch" ; 
    const char* m_tag = "-4" ;
    const char* m_udet = "PmtInBox" ; 
    
    //const char* m_typ = "G4Gun" ; 
    //const char* m_tag = "-1" ;
    //const char* m_udet = "G4Gun" ; 

    std::string ixdir = NPYBase::directory( "ix", m_typ, m_udet );  

    std::cout << argv[0] << " ixdir " << ixdir << std::endl ;  

    if(1)
    {
        Index* seqhis = Index::load(ixdir.c_str(), m_tag, "History_Sequence" );  // SEQHIS_NAME_
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
    }
    if(0)
    {
        Index* seqmat = Index::load(ixdir.c_str(), m_tag, "Material_Sequence");  // SEQMAT_NAME_
        GItemIndex* m_seqmat = new GItemIndex(seqmat);
        m_seqmat->setTypes(&types);
        m_seqmat->setLabeller(GItemIndex::MATERIALSEQ);
        m_seqmat->formTable();
        dump(m_seqmat, "m_seqmat");
    }
    if(0)
    {
        Index* bndidx_ = Index::load(ixdir.c_str(), m_tag, "Boundary_Index");     // BNDIDX_NAME_
        GItemIndex* bndidx = new GItemIndex(bndidx_);
        bndidx->dump("bndidx");
    }


    return 0 ;
}
