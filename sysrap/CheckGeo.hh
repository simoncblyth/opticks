#include "SGeo.hh"

#include "SYSRAP_API_EXPORT.hh"

struct sframe ; 

struct SYSRAP_API CheckGeo : public SGeo 
{
    unsigned           getNumMeshes() const ; 
    const char*        getMeshName(unsigned midx) const ;
    int                getMeshIndexWithName(const char* name, bool startswith) const ;
    int                getFrame(sframe& fr, int ins_idx ) const ; 
    std::string        descBase() const ; 
    int                lookup_mtline(int mtindex) const ; 
};



