#include "GSurfaceIndex.hh"

GSurfaceIndex* GSurfaceIndex::load(const char* idpath)
{
    GSurfaceIndex* gsi = new GSurfaceIndex ;  
    gsi->loadMaps(idpath);
    return gsi ; 
}


