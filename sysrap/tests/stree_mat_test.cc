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


    NPFold* fold = new NPFold ; 

    // arrays directly under SSim all from old X4/GGeo workflow
    const NP* _oldmat = NP::Load(base, "oldmat.npy"); 
    const NP* _oldsur = NP::Load(base, "oldsur.npy"); 
    const NP* _oldbnd = NP::Load(base, "bnd.npy"); 
    const NP* _oldoptical = NP::Load(base, "optical.npy"); 

    fold->add("oldmat", _oldmat ); 
    fold->add("oldsur", _oldsur ); 
    fold->add("oldbnd", _oldbnd ); 
    fold->add("oldoptical", _oldoptical ); 

    // U4Material::MakeStandardArray, U4Material::MakeStandardSurface
    fold->add("mat", st.mat ); 
    fold->add("sur", st.sur ); 
    fold->add("bnd", st.make_bnd() ); 
    fold->add("bd",  st.make_bd()  ); 
    fold->add("optical",  st.make_optical()  ); 

    fold->add("rayleigh",  st.rayleigh  ); 
    fold->add("energy",  st.energy ); 
    fold->add("wavelength",  st.wavelength ); 

    fold->save("$FOLD"); 
 
    return 0 ; 
}
