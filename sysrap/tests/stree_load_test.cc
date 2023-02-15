#include "ssys.h"
#include "stree.h"
#include "snd.hh"

const char* BASE = getenv("BASE");  
const int LVID = ssys::getenvint("LVID", -1); 
const int NDID = ssys::getenvint("NDID",  0); 

void test_get_combined_transform( const stree& st )
{
    std::cout 
        << "test_get_combined_transform" 
        << " LVID " << LVID 
        << " NDID " << NDID 
        << std::endl 
        ; 

    std::vector<snode> nodes ;  // Volume nodes with the LV
    st.find_lvid_nodes_(nodes, LVID) ; 

    int num_nodes = nodes.size(); 
    std::cout << " VOL " << st.desc_nodes_(nodes) ; 

    std::vector<snd> nds ;    // CSG constituent nodes of the LV 
    snd::GetLVID(nds, LVID ); 
    int num_nds = nds.size(); 

    std::cout << " CSG " << snd::Brief_(nds) ; 

    assert( num_nodes > 0 && num_nds > 1 ); 

    std::vector<glm::tmat4x4<double>> tvs ;
    tvs.reserve(num_nodes*2);  

    for(int i=0 ; i < num_nodes ; i++)
    {
        bool dump_NDID = i == NDID ;  

        const snode& node = nodes[i] ; 
        const snd* nd = &nds[0] ; 

        glm::tmat4x4<double> t(1.) ; 
        glm::tmat4x4<double> v(1.) ; 

        std::stringstream* out = dump_NDID ? new std::stringstream : nullptr ; 

        st.get_combined_transform(t, v, node, nd, out ); 

        tvs.push_back(t); 
        tvs.push_back(v);
 
        if(out) 
        {
            std::string str = out->str(); 
            std::cout 
                << " dump_NDID " << ( dump_NDID ? "YES" : "NO" )
                << " i " << std::setw(3) << i 
                << std::endl 
                << stra<double>::Desc(t, v, "t", "v") 
                << std::endl 
                << str 
                << std::endl 
                ; 
        }
    }

    NP* a = NP::Make<double>( num_nodes, 2, 4, 4); 
    a->read2<double>( (double*)tvs.data() ); 

    const char* path = "/tmp/test_get_combined_transform.npy" ; 
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
        test_get_combined_transform(st);  
    }
    else
    {
        std::cout << st.desc() ; 
    }


    return 0 ; 
}
