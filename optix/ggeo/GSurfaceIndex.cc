#include "GSurfaceIndex.hh"


GSurfaceIndex* GSurfaceIndex::load(const char* idpath)
{
    GSurfaceIndex* gsi = new GSurfaceIndex ;  
    gsi->loadIndex(idpath);
    return gsi ; 
}

GSurfaceIndex::GSurfaceIndex() : GItemIndex("GSurfaceIndex")
{
}

