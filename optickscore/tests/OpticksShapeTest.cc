#include "OPTICKS_LOG.hh"
#include "OpticksShape.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    
    LOG(info) << argv[0] ; 

    unsigned meshIdx  = 0xdead ; 
    unsigned boundaryIdx = 0xbeef ; 

    unsigned shape = OpticksShape::Encode(meshIdx, boundaryIdx); 

    unsigned meshIdx2 = OpticksShape::MeshIndex(shape); 
    unsigned boundaryIdx2 = OpticksShape::BoundaryIndex(shape); 

    assert( meshIdx2 == meshIdx ); 
    assert( boundaryIdx2 == boundaryIdx ); 

    glm::uvec4 id(0,0,shape,0); 
    unsigned meshIdx3 = OpticksShape::MeshIndex(id); 
    unsigned boundaryIdx3 = OpticksShape::BoundaryIndex(id); 

    assert( meshIdx3 == meshIdx ); 
    assert( boundaryIdx3 == boundaryIdx ); 

    return 0 ;
}

// om-;TEST=OpticksShapeTest om-t
