#include "NP.hh"
#include "SBnd.h"
#include "stree.h"

int main(int argc, char** argv)
{
    const char* specs_ = R"(
Red///Green
Green///Blue
Blue///Cyan
Cyan///Magenta
Magenta///Yellow
)"; 

    std::vector<std::string> specs ; 
    SBnd::GetSpecsFromString(specs, specs_ ); 

    NP* bnd = NP::MakeFlat<float>( specs.size(), 4, 2, 761, 4 ); 
    bnd->set_names(specs); 

    stree st ; 

    st.add_material("Red",   0u) ; 
    st.add_material("Green", 1u) ; 
    st.add_material("Blue",  2u) ; 
    st.add_material("Cyan",  3u) ; 
    st.add_material("Magenta", 4u) ; 
    st.add_material("Yellow", 5u) ; 

    SBnd sbn(bnd); 
    sbn.fillMaterialLine( &st ); 
    std::cout << st.desc_mt() << std::endl ; 

    st.init_mtindex_to_mtline(); 



    for(unsigned i=0 ; i < st.mtindex.size() ; i++)
    {
        int mtindex = st.mtindex[i] ;
        //int mtline = st.mtindex_to_mtline.at(mtindex) ;
        int mtline = st.lookup_mtline(mtindex); 

        std::cout 
            << " mtindex " << std::setw(3) << mtindex 
            << " mtline " << std::setw(3) << mtline 
            << std::endl
            ; 
    }    
    for(unsigned i=0 ; i < 10 ; i++)
    {
        int mtindex = i ;
        int mtline = st.lookup_mtline(i); 
        std::cout 
            << " mtindex " << std::setw(3) << mtindex 
            << " mtline " << std::setw(3) << mtline 
            << std::endl
            ; 
    }

    std::cout << sbn.desc() << std::endl ; 

    return 0 ; 
}
