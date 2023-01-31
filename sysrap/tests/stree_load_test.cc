#include "ssys.h"
#include "stree.h"
#include "snd.hh"

const char* BASE = getenv("BASE");  
const int LVID = ssys::getenvint("LVID", -1); 

int main(int argc, char** argv)
{
    stree st ; 
    st.load(BASE); 
    
    if( LVID > 0 )
    {
        std::vector<snd> nds ; 
        snd::GetLVID(nds, LVID ); 

        int num_nds = nds.size(); 

        std::cout << " LVID " << LVID << " num_nds " << num_nds << std::endl ; 
        for(int i=0 ; i < num_nds ; i++) std::cout << nds[i].brief() << std::endl ; 
        //for(int i=0 ; i < num_nds ; i++) std::cout << nds[i].desc() << std::endl ; 

    }
    else
    {
        std::cout << st.desc() ; 
    }


    return 0 ; 
}
