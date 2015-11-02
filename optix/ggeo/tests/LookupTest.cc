#include "GCache.hh"
#include "GBndLib.hh"
#include "Lookup.hpp"


int main()
{
    GCache* m_cache = new GCache("GGEOVIEW_");

    GBndLib* blib = GBndLib::load(m_cache, true );
    blib->dump();



    Lookup* m_lookup = new Lookup();

    m_lookup->loadA( m_cache->getIdFold(), "ChromaMaterialMap.json", "/dd/Materials/") ;

    blib->fillMaterialLineMap( m_lookup->getB() ) ;    

    m_lookup->crossReference();

    m_lookup->dump("ggeo-/LookupTest");




    printf("  a => b \n");
    for(unsigned int a=0; a < 35 ; a++ )
    {   
        int b = m_lookup->a2b(a);
        std::string aname = m_lookup->acode2name(a) ;
        std::string bname = m_lookup->bcode2name(b) ;
        printf("  %3u -> %3d  ", a, b );

        if(b < 0) printf(" %25s : WARNING failed to translate acode %u \n", aname.c_str(), a);    
        else
        {   
             assert(aname == bname);
             printf(" %25s \n", aname.c_str() );
        }   
    }   

    


}
