// op --gitemindex 

#include <string>
#include <iostream>

// ggeo-
#include "GItemIndex.hh"

// npy-
#include "Types.hpp"
#include "Index.hpp"
#include "NumpyEvt.hpp"

void dump(GItemIndex* idx, const char* msg)
{
    std::cout << idx->gui_radio_select_debug();
    idx->test();
    idx->dump(msg);
}

int main(int argc, char** argv)
{
    Types types ; 

    const char* m_typ = "torch" ; 
    const char* m_tag = "-4" ;
    const char* m_udet = "PmtInBox" ; 
    
    //const char* m_typ = "G4Gun" ; 
    //const char* m_tag = "-1" ;
    //const char* m_udet = "G4Gun" ; 

    std::string ixdir = NumpyEvt::speciesDir( "ix", m_udet, m_typ );  
    std::cout << argv[0] << " ixdir " << ixdir << std::endl ;  

    if(1)
    {
        Index* seqhis = Index::load(ixdir.c_str(), m_tag, "History_Sequence" );  // SEQHIS_NAME_
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
