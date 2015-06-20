#include "GMaterialIndex.hh"


GMaterialIndex* GMaterialIndex::load(const char* idpath)
{
    GMaterialIndex* gmi = new GMaterialIndex ;  
    gmi->loadMaps(idpath);
    return gmi ; 
}




