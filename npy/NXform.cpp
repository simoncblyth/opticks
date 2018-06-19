#include "NXform.hpp"
#include "NPY.hpp"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NGLMCF.hpp"
#include "PLOG.hh"


nxform::nxform(unsigned num_nodes_, bool debug_)
    :
    num_nodes(num_nodes_),
    debug(debug_),
    num_triple(0),
    num_triple_mismatch(0),
    triple(debug_ ? NPY<float>::make( num_nodes, 3, 4, 4 ) : NULL )
{
} 


const nmat4triple* nxform::make_triple( const float* data)
{
    // spell out nglmext::invert_trs for debugging discrepancies

    glm::mat4 T = glm::make_mat4(data) ;
    ndeco d = nglmext::polar_decomposition( T ) ; 

    glm::mat4 isirit = d.isirit ;
    glm::mat4 i_trs = glm::inverse( T ) ;

    NGLMCF cf(isirit, i_trs );

    if(!cf.match)
    {
        num_triple_mismatch++ ;
        //LOG(warning) << cf.desc("nd::make_triple polar_decomposition inverse and straight inverse are mismatched " );
    }

    glm::mat4 V = isirit ;
    glm::mat4 Q = glm::transpose(V) ;

    nmat4triple* tvq = new nmat4triple(T, V, Q);

    if(triple)  // collecting triples for mismatch debugging 
    {
        triple->setMat4Triple( tvq , num_triple++ );
    }
    return tvq ;
}



// node structs that can work with this require
// transform and parent members   

template <typename N>
const nmat4triple* nxform::make_global_transform(const N* n) // static
{
    std::vector<const nmat4triple*> tvq ; 
    while(n)
    {
        if(n->transform) tvq.push_back(n->transform);
        n = n->parent ; 
    }
    bool reverse = true ; // as tvq in leaf-to-root order
    return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ; 
}




