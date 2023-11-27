#include "CheckGeo.hh"
#include "sframe.h"


unsigned CheckGeo::getNumMeshes() const 
{
    return 42 ; 
}
const char* CheckGeo::getMeshName(unsigned ) const 
{
    return nullptr ; 
}
int CheckGeo::getMeshIndexWithName(const char* , bool ) const 
{
   return 0 ; 
}




int CheckGeo::getFrame(sframe& fr, int ins_idx ) const 
{
    fr.zero(); 
    return 0 ; 
}

std::string CheckGeo::descBase() const 
{
    return "CheckGeo::descBase" ; 
}

int CheckGeo::lookup_mtline(int mtindex) const 
{
    return -1 ; 
}
std::string CheckGeo::desc_mt() const 
{
    return "CheckGeo::desc_mt"  ; 
}


