#include "GMaterialIndex.hh"


GMaterialIndex* GMaterialIndex::load(const char* idpath)
{
    GMaterialIndex* gmi = new GMaterialIndex ;    // itemname->index
    gmi->loadIndex(idpath);
    return gmi ; 
}




