
#include "NQuad.hpp"
#include "NFieldGrid3.hpp"
#include "NField3.hpp"
#include "NGrid3.hpp"


NFieldGrid3::NFieldGrid3(NField3* field, NGrid3* grid, bool offset)      
    : 
    field(field),
    grid(grid),
    offset(offset)
{
}

/*
void NFieldGrid3::setGridOffset(NGrid3* grid, bool offset)
{
    m_grid = grid ; 
    m_offset = offset ; 
}
*/


nvec3 NFieldGrid3::position_f( const nvec3& ijkf_ ) const 
{
    nvec3 ijkf = ijkf_ ; 
    if(offset) ijkf -= make_nvec3(grid->half_min) ; 

    nvec3 frac_pos = grid->fpos(ijkf);
    nvec3 world_pos = field->position(frac_pos);
    return world_pos ; 
}

nvec3 NFieldGrid3::position( const nivec3& ijk_ ) const 
{
    nivec3 ijk = ijk_ ; 
    if(offset) ijk -= grid->half_min ; 

    nvec3 frac_pos = grid->fpos(ijk);
    nvec3 world_pos = field->position(frac_pos);
    return world_pos ; 
}



float NFieldGrid3::value( const nivec3& ijk_ ) const 
{
    nivec3 ijk = ijk_ ; 
    if(offset) ijk -= grid->half_min ; 

    nvec3 frac_pos = grid->fpos(ijk);
    float vfi = (*field)(frac_pos) ; 

    return vfi  ; 
}

float NFieldGrid3::value_f( const nvec3& ijkf_ ) const 
{
    nvec3 ijkf = ijkf_ ; 
    if(offset) ijkf -= make_nvec3(grid->half_min) ; 

    nvec3 frac_pos = grid->fpos(ijkf);
    float vfi = (*field)(frac_pos) ; 

    return vfi  ; 
}

float NFieldGrid3::value_f( const glm::vec3& ijkf_ ) const 
{
    glm::vec3 ijkf = ijkf_ ; 
    if(offset) ijkf -= glm::vec3(grid->half_min.x, grid->half_min.y, grid->half_min.z) ; 

    nvec3 frac_pos = grid->fpos(ijkf);
    float vfi = (*field)(frac_pos) ; 

    return vfi  ; 
}

