#include "stree.h"

int main(int argc, char** argv)
{
    stree st ; 
    int rc = st.load("$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim"); 
    if( rc != 0 ) return rc ; 

    
    //std::cout << "st.desc" << std::endl << st.desc() << std::endl ; 

    std::cout << "st.material.desc" << std::endl << st.material->desc() << std::endl ; 

    std::cout << "st.desc_mt" << std::endl << st.desc_mt() << std::endl; 

    std::cout << "st.desc_bd" << std::endl << st.desc_bd() << std::endl; 


    NP* mat = st.create_mat(); 
    assert( mat != nullptr ); 
    mat->save("$FOLD/mat.npy"); 
 
    return 0 ; 
}
