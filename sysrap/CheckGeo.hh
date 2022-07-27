#include "SGeo.hh"

#include "SYSRAP_API_EXPORT.hh"

struct sframe ; 

class SYSRAP_API CheckGeo : public SGeo 
{
    public:
        unsigned           getNumMeshes() const ; 
        const char*        getMeshName(unsigned midx) const ;
        int                getMeshIndexWithName(const char* name, bool startswith) const ;
        int                getFrame(sframe& fr, int ins_idx ) const ; 
        std::string        descBase() const ; 

};



