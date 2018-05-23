#include "NGLMExt.hpp"

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
FVec NFieldGrid3<FVec,IVec>::position_f( const FVec& ijkf_, bool debug ) const 
{
   // bizarrely this is seeing input as integral  

    FVec ijkf = ijkf_ ; 
    if(offset) ijkf -= grid->half_min ; 

    FVec frac_pos = grid->fpos(ijkf, debug);
    FVec world_pos = field->position(frac_pos);

    if(debug) std::cout << "NFieldGrid3::position_f"
                        << " ijkf " << ijkf
                        << " frac_pos " << frac_pos 
                        << " world_pos " << world_pos
                        << std::endl ; 
                
    return world_pos ; 
}



/*
template<typename FVec, typename IVec>
FVec NFieldGrid3<FVec,IVec>::position_f( float i, float j, float k, bool debug ) const 
{
    glm::vec3 ijkf(i,j,k) ; 

    if(debug) std::cout << "NFieldGrid3::position_f"
                        << " i " << i
                        << " j " << j
                        << " k " << k
                        << " ijkf.x " << ijkf.x
                        << " ijkf.y " << ijkf.y
                        << " ijkf.z " << ijkf.z
                        << std::endl ; 

    return position_f(ijkf, debug);
}

*/




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

