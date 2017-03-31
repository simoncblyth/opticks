
#include "NQuad.hpp"
#include "NFieldGrid3.hpp"
#include "NField3.hpp"
#include "NGrid3.hpp"


template<typename FVec, typename IVec>
NFieldGrid3<FVec,IVec>::NFieldGrid3(NField<FVec,IVec,3>* field, NGrid<FVec,IVec,3>* grid, bool offset)      
    : 
    field(field),
    grid(grid),
    offset(offset)
{
}


template<typename FVec, typename IVec>
FVec NFieldGrid3<FVec,IVec>::position_f( const FVec& ijkf_ ) const 
{
    FVec ijkf = ijkf_ ; 
    if(offset) ijkf -= grid->half_min ; 

    FVec frac_pos = grid->fpos(ijkf);
    FVec world_pos = field->position(frac_pos);
    return world_pos ; 
}

template<typename FVec, typename IVec>
FVec NFieldGrid3<FVec,IVec>::position( const IVec& ijk_ ) const 
{
    IVec ijk = ijk_ ; 
    if(offset) ijk -= grid->half_min ; 

    FVec frac_pos = grid->fpos(ijk);
    FVec world_pos = field->position(frac_pos);
    return world_pos ; 
}


template<typename FVec, typename IVec>
float NFieldGrid3<FVec,IVec>::value( const IVec& ijk_ ) const 
{
    IVec ijk = ijk_ ; 
    if(offset) ijk -= grid->half_min ; 

    FVec frac_pos = grid->fpos(ijk);
    float vfi = (*field)(frac_pos) ; 

    return vfi  ; 
}

template<typename FVec, typename IVec>
float NFieldGrid3<FVec,IVec>::value_f( const FVec& ijkf_ ) const 
{
    FVec ijkf = ijkf_ ; 
    if(offset) ijkf -= grid->half_min ; 

    FVec frac_pos = grid->fpos(ijkf);
    float vfi = (*field)(frac_pos) ; 

    return vfi  ; 
}



template struct NPY_API NFieldGrid3<glm::vec3, glm::ivec3> ; 

