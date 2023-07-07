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
    const NP* _oldbnd = NP::Load(ssbase, "bnd.npy"); 
    const NP* _oldoptical = NP::Load(ssbase, "optical.npy"); 


    // st.postinit();  // abnormal call : prior to remaking GEOM 


    NPFold* fold = new NPFold ; 

    fold->add("oldmat", _oldmat ); 
    fold->add("oldsur", _oldsur ); 
    fold->add("oldbnd", _oldbnd ); 
    fold->add("oldoptical", _oldoptical ); 



    // TODO: group these 8 into fold populated by stree::save 
    // then can eliminate this executable can just 
    // directly load from the persisted GEOM 
 
    fold->add("mat", st.mat ); 
    fold->add("sur", st.sur ); 
    fold->add("rayleigh",  st.rayleigh  ); 
    fold->add("energy",  st.energy ); 
    fold->add("wavelength",  st.wavelength ); 
    fold->add("bnd", st.bnd ); 
    fold->add("bd",  st.bd  ); 
    fold->add("optical",  st.optical  ); 



    fold->save("$FOLD"); 
 
    return 0 ; 
}
