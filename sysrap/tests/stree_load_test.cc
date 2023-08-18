#include "ssys.h"
#include "stree.h"

#ifdef WITH_SND
#include "snd.hh"
#else
#include "sn.h"
#endif

void test_get_combined_transform( const stree& st, int LVID, int NDID )
{
    std::cout 
        << "test_get_combined_transform" 
#ifdef WITH_SND
        << " WITH_SND " 
#else
        << " NOT:WITH_SND "
#endif
        << " LVID " << LVID
        << " NDID " << NDID 
        << std::endl 
        ; 

    std::vector<snode> nodes ;  // structural volume nodes with the LVID, could be thousands
    st.find_lvid_nodes_(nodes, LVID) ; 

    int num_nodes = nodes.size(); 
    std::cout << " VOL " << st.desc_nodes_(nodes) ; 


#ifdef WITH_SND
    std::vector<snd> nds ;    // CSG constituent nodes of the LV 
    snd::GetLVID(nds, LVID ); 
    int num_nds = nds.size(); 
    std::cout << " snd::Brief_(nds) " << std::endl << snd::Brief_(nds) ; 
#else
    std::vector<sn*> nds ;    // CSG constituent nodes of the LV 
    sn::GetLVNodes(nds, LVID ); 
    int num_nds = nds.size(); 
    std::cout << " sn::Desc(nds) " << std::endl << sn::Desc(nds) ; 
#endif

    assert( num_nodes > 0 && num_nds > 1 ); 

    std::vector<glm::tmat4x4<double>> tvs ;
    tvs.reserve(num_nodes*2);  

    for(int i=0 ; i < num_nodes ; i++)
    {
        bool dump_NDID = i == NDID ;  

        const snode& node = nodes[i] ; 
#ifdef WITH_SND
        const snd* nd = &nds[0] ; 
#else
        const sn*  nd = nds[0] ; 
#endif

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

    //NP* a = NPX::ArrayFromVector<double>( tvs, 2, 4, 4) ; 


    const char* path = "/tmp/test_get_combined_transform.npy" ; 
    std::cout << " save " << path << std::endl ; 
    a->save(path); 
}



int main(int argc, char** argv)
{
    stree st ; 
    int rc = st.load("$BASE"); 
    if( rc != 0 ) return rc ; 

    int LVID = ssys::getenvint("LVID",  0); 
    int NDID = ssys::getenvint("NDID",  0); 
    
    std::cout << st.desc() ; 
    test_get_combined_transform(st, LVID, NDID );  


    return 0 ; 
}
