#include "ssys.h"
#include "stree.h"
#include "stra.h"
#include "strid.h"

#ifdef WITH_SND
#include "snd.hh"
#else
#include "sn.h"
#endif


struct stree_load_test
{ 
    const stree* st ; 
    stree_load_test(const stree* st_);

    void get_combined_transform(int LVID, int NDID  );  
    void get_inst(int idx) const ; 
    void get_instance_frame(int idx) const ; 

};

inline stree_load_test::stree_load_test(const stree* st_) 
    :
    st(st_)
{
}


inline void stree_load_test::get_combined_transform( int LVID, int NDID )
{
    std::cout 
        << "stree_load_test::get_combined_transform" 
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
    st->find_lvid_nodes_(nodes, LVID) ; 

    int num_nodes = nodes.size(); 
    std::cout << " VOL " << st->desc_nodes_(nodes) ; 


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

        st->get_combined_transform(t, v, node, nd, out ); 

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

inline void stree_load_test::get_inst(int idx) const 
{
    const glm::tmat4x4<double>* inst = st->get_inst(idx) ; 
    const glm::tmat4x4<double>* iinst = st->get_iinst(idx) ; 
    const glm::tmat4x4<float>* inst_f4 = st->get_inst_f4(idx) ; 
    const glm::tmat4x4<float>* iinst_f4 = st->get_iinst_f4(idx) ; 

    std::cout 
        << "stree_load_test::get_inst"
        << " idx " << idx 
        << " inst " << ( inst ? "YES" : "NO " )
        << " iinst " << ( iinst ? "YES" : "NO " )
        << " inst_f4 " << ( inst_f4 ? "YES" : "NO " )
        << " iinst_f4 " << ( iinst_f4 ? "YES" : "NO " )
        << std::endl 
        ;

    if(inst)  std::cout << "inst"  << std::endl << stra<double>::Desc(*inst) << std::endl ; 
    if(iinst) std::cout << "iinst" << std::endl << stra<double>::Desc(*iinst) << std::endl ; 
    if(inst_f4)  std::cout << "inst_f4"  << std::endl << stra<float>::Desc(*inst_f4) << std::endl ; 
    if(iinst_f4) std::cout << "iinst_f4" << std::endl << stra<float>::Desc(*iinst_f4) << std::endl ; 

    if(inst)  std::cout << "inst"  << std::endl << strid::Desc<double,int64_t>(*inst) << std::endl ; 
    if(iinst) std::cout << "iinst" << std::endl << strid::Desc<double,int64_t>(*iinst) << std::endl ; 
    if(inst_f4)  std::cout << "inst_f4"  << std::endl << strid::Desc<float,int32_t>(*inst_f4) << std::endl ; 
    if(iinst_f4) std::cout << "iinst_f4" << std::endl << strid::Desc<float,int32_t>(*iinst_f4) << std::endl ; 



    

}


/**
HMM: sframe is qat4 (float) based, need templated ? 
**/
inline void stree_load_test::get_instance_frame(int idx) const 
{
}




int main(int argc, char** argv)
{
    stree st ; 
    int rc = st.load("$BASE"); 
    if( rc != 0 ) return rc ; 
    std::cout << st.desc() ; 

    int LVID = ssys::getenvint("LVID",  0); 
    int NDID = ssys::getenvint("NDID",  0); 
    int IIDX = ssys::getenvint("IIDX",  0); 

    stree_load_test test(&st); 
    //test.get_combined_transform(LVID, NDID );  
    test.get_inst(IIDX) ; 

    return 0 ; 
}
