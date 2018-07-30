#include "OPTICKS_LOG.hh"
#include "NPYBase.hpp"
#include "NPYSpec.hpp"
#include "NPYSpecList.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    typedef enum 
    { 
       SRC_NODES,
       SRC_IDX, 
       SRC_TRANSFORMS, 
       SRC_PLANES, 
       SRC_FACES, 
       SRC_VERTS,

       NODES,
       TRANSFORMS, 
       GTRANSFORMS, 
       IDX

    }  NCSGData_t ; 

    const NPYSpec* SRC_NODES_      = new NPYSpec("srcnodes.npy"       , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) ; 
    const NPYSpec* SRC_IDX_        = new NPYSpec("srcidx.npy"         , 0, 4, 0, 0, 0, NPYBase::UINT  , "" ) ; 
    const NPYSpec* SRC_TRANSFORMS_ = new NPYSpec("srctransforms.npy"  , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) ; 
    const NPYSpec* SRC_PLANES_     = new NPYSpec("srcplanes.npy"      , 0, 4, 0, 0, 0, NPYBase::FLOAT , "" ) ; 
    const NPYSpec* SRC_FACES_      = new NPYSpec("srcfaces.npy"       , 0, 3, 0, 0, 0, NPYBase::INT   , "" ) ; 
    const NPYSpec* SRC_VERTS_      = new NPYSpec("srcverts.npy"       , 0, 3, 0, 0, 0, NPYBase::FLOAT , "" ) ; 
    const NPYSpec* NODES_          = new NPYSpec("nodes.npy"          , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) ; 
    const NPYSpec* TRANSFORMS_     = new NPYSpec("transforms.npy"     , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) ; 
    const NPYSpec* GTRANSFORMS_    = new NPYSpec("gtransforms.npy"    , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) ; 
    const NPYSpec* IDX_            = new NPYSpec("idx.npy"            , 0, 4, 0, 0, 0, NPYBase::UINT  , "" ) ; 

    NPYSpecList sl ; 

    sl.add( (unsigned)SRC_NODES       , SRC_NODES_ );
    sl.add( (unsigned)SRC_IDX         , SRC_IDX_ );
    sl.add( (unsigned)SRC_TRANSFORMS  , SRC_TRANSFORMS_ );
    sl.add( (unsigned)SRC_PLANES      , SRC_PLANES_ );
    sl.add( (unsigned)SRC_FACES       , SRC_FACES_ );
    sl.add( (unsigned)SRC_VERTS       , SRC_VERTS_ );
    sl.add( (unsigned)NODES           , NODES_ );
    sl.add( (unsigned)TRANSFORMS      , TRANSFORMS_ );
    sl.add( (unsigned)GTRANSFORMS     , GTRANSFORMS_ );
    sl.add( (unsigned)IDX             , IDX_ );

    LOG(info) << std::endl << sl.description() ; 

    NCSGData_t bid = GTRANSFORMS ; 

    const NPYSpec* spec = sl.getByIdx( (unsigned)bid ); 
    LOG(info) << " GTRANSFORMS : " << spec->description() ; 


    return 0 ; 
}


