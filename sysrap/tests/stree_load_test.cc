#include "ssys.h"
#include "stree.h"
#include "snd.hh"

const char* BASE = getenv("BASE");  
const int LVID = ssys::getenvint("LVID", -1); 

void test_get_transform( const stree& st )
{
    std::cout << "test_get_transform LVID " << LVID << std::endl ; 
    std::vector<snode> nodes ;  // Volume nodes with the LV
    st.find_lvid_nodes_(nodes, LVID) ; 

    int num_nodes = nodes.size(); 
    std::cout << " VOL " << st.desc_nodes_(nodes) ; 

    std::vector<snd> nds ;    // CSG constituent nodes of the LV 
    snd::GetLVID(nds, LVID ); 
    int num_nds = nds.size(); 

    std::cout << " CSG " << snd::Brief_(nds) ; 

    assert( num_nodes > 0 && num_nds > 1 ); 

    std::vector<glm::tmat4x4<double>> trs ;
    trs.reserve(num_nodes);  

    for(int i=0 ; i < num_nodes ; i++)
    {
        const snode& node = nodes[i] ; 
        const snd* nd = &nds[0] ; 

        glm::tmat4x4<double> tr(1.) ; 
        st.get_transform(tr, node, nd); 
        trs.push_back(tr); 
        std::cout << " i " << std::setw(3) << i << " tr " << glm::to_string(tr) << std::endl ; 
    }

    NP* a = NP::Make<double>( num_nodes, 4, 4); 
    a->read2<double>( (double*)trs.data() ); 

    const char* path = "/tmp/test_get_transform.npy" ; 
    std::cout << " save " << path << std::endl ; 
    a->save(path); 
}



int main(int argc, char** argv)
{
    stree st ; 
    int rc = st.load(BASE); 
    if( rc != 0 ) return rc ; 
    
    if( LVID > 0 )
    {
        test_get_transform(st);  
    }
    else
    {
        std::cout << st.desc() ; 
    }


    return 0 ; 
}
