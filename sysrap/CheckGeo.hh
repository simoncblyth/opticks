#include "SGeo.hh"

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API CheckGeo : public SGeo 
{
    public:
        unsigned           getNumMeshes() const ; 
        const char*        getMeshName(unsigned midx) const ;
        int                getMeshIndexWithName(const char* name, bool startswith) const ;

        //CheckGeo(); 
        //virtual ~CheckGeo() ; 

};



