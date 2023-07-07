#include "stree.h"
#include "NPX.h"

int main(int argc, char** argv)
{
    const char* ssbase = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ; 

    stree st ; 
    int rc = st.load(ssbase); 
    if( rc != 0 ) return rc ; 

    
    //std::cout << "st.desc" << std::endl << st.desc() << std::endl ; 

    std::cout << "st.material.desc" << std::endl << st.material->desc() << std::endl ; 
    std::cout << "st.desc_mt" << std::endl << st.desc_mt() << std::endl; 

    std::cout << "st.surface.desc" << std::endl << st.surface->desc() << std::endl ; 
    std::cout << "st.desc_bd" << std::endl << st.desc_bd() << std::endl; 


    // arrays directly under SSim all from old X4/GGeo workflow
    const NP* _oldmat = NP::Load(ssbase, "oldmat.npy"); 
    const NP* _oldsur = NP::Load(ssbase, "oldsur.npy"); 
    const NP* _oldbnd = NP::Load(ssbase, "oldbnd.npy"); 
    const NP* _oldoptical = NP::Load(ssbase, "oldoptical.npy"); 



    NPFold* fold = new NPFold ; 

    fold->add("oldmat", _oldmat ); 
    fold->add("oldsur", _oldsur ); 
    fold->add("oldbnd", _oldbnd ); 
    fold->add("oldoptical", _oldoptical ); 


    fold->add("mat",         st.standard->mat ); 
    fold->add("sur",         st.standard->sur ); 
    fold->add("rayleigh",    st.standard->rayleigh  ); 
    fold->add("energy",      st.standard->energy ); 
    fold->add("wavelength",  st.standard->wavelength ); 
    fold->add("bnd",         st.standard->bnd ); 
    fold->add("bd",          st.standard->bd  ); 
    fold->add("optical",     st.standard->optical  ); 



    fold->save("$FOLD"); 
 
    return 0 ; 
}
