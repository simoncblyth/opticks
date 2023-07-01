#include "stree.h"
#include "NPX.h"

int main(int argc, char** argv)
{
    const char* base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ; 

    stree st ; 
    int rc = st.load(base); 
    if( rc != 0 ) return rc ; 

    
    //std::cout << "st.desc" << std::endl << st.desc() << std::endl ; 

    std::cout << "st.material.desc" << std::endl << st.material->desc() << std::endl ; 

    std::cout << "st.desc_mt" << std::endl << st.desc_mt() << std::endl; 


    std::cout << "st.surface.desc" << std::endl << st.surface->desc() << std::endl ; 

    std::cout << "st.desc_bd" << std::endl << st.desc_bd() << std::endl; 


    const NP* _mat = st.create_mat() ; 
    const NP* _sur = st.create_sur() ; 
    const NP* _bnd = st.create_bnd(_mat, _sur); 

    const NP* _old_bnd = NP::Load(base, "bnd.npy"); 
    const NP* _old_optical = NP::Load(base, "optical.npy"); 

    NP* _bd = NPX::ArrayFromVec<int,int4>(st.bd, 4) ; 
    _bd->set_names(st.bdname) ; 


    NPFold* fold = new NPFold ; 
    fold->add("mat", _mat ); 
    fold->add("sur", _sur ); 
    fold->add("bnd", _bnd ); 
    fold->add("bd",  _bd ); 


    fold->add("old_bnd", _old_bnd ); 
    fold->add("old_optical", _old_optical ); 
    fold->save("$FOLD"); 

 
    return 0 ; 
}
