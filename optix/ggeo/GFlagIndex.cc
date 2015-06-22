#include "GFlagIndex.hh"

#include "regexsearch.hh"


GFlagIndex* GFlagIndex::load(const char* idpath)
{
    GFlagIndex* gfi = new GFlagIndex ;    // itemname->index
 
    typedef std::pair<unsigned int, std::string>  upair_t ;
    typedef std::vector<upair_t>                  upairs_t ;
    upairs_t ups ; 
    enum_regexsearch( ups, "$ENV_HOME/graphics/ggeoview/cu/photon.h");    

    for(unsigned int i=0 ; i < ups.size() ; i++)
    {
        upair_t p = ups[i];
        gfi->add( p.second.c_str(), ffs(p.first) ); 
    }
    //gfi->formTable(); cannot do here as no colors/mappings yet
   
    return gfi ; 
}




