//  ggv --attr

#include "GCache.hh"
#include "GFlags.hh"
#include "GMaterialLib.hh"
#include "GAttrSeq.hh"
#include "Index.hpp"

#include <iostream>
#include <iomanip>

void test_history_sequence(GCache* cache)
{
    GFlags* flags = cache->getFlags();
    GAttrSeq* qflg = flags->getAttrIndex();
    qflg->dump();

    Index* seqhis = Index::load(cache->getIdPath(), "History_Sequence");
    seqhis->dump();

    for(unsigned int i=0 ; i < seqhis->getNumKeys() ; i++)
    {
        const char* key = seqhis->getKey(i);
        assert(key);
        //if(!key) continue ; 

        unsigned int count = seqhis->getIndexSource(key);

        std::string dseq = qflg->decodeHexSequenceString(key);

        std::cout << std::setw(5) << i 
                  << std::setw(10) << count 
                  << std::setw(25) << ( key ? key : "-" ) 
                  << std::setw(40) << dseq 
                  << std::endl ;
    }
}

void test_material_sequence(GCache* cache)
{
    GMaterialLib* mlib = GMaterialLib::load(cache);
    GAttrSeq* qmat = mlib->getAttrNames();
    qmat->dump();
    //const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;

    Index* seqmat = Index::load(cache->getIdPath(), "Material_Sequence");
    seqmat->dump();

    for(unsigned int i=0 ; i < seqmat->getNumKeys() ; i++)
    {
        const char* key = seqmat->getKey(i);
        assert(key);
        //if(!key) continue ; 

        unsigned int count = seqmat->getIndexSource(key);

        std::string dseq = qmat->decodeHexSequenceString(key);

        std::cout << std::setw(5) << i 
                  << std::setw(10) << count 
                  << std::setw(25) << ( key ? key : "-" ) 
                  << std::setw(40) << dseq 
                  << std::endl ;
    }



}







int main()
{
    GCache gc("GGEOVIEW_");

    test_history_sequence(&gc);

    test_material_sequence(&gc);

}
