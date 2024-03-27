/**
stree_load_test.cc
===================

::

    ~/o/sysrap/tests/stree_load_test.sh  

    TEST=pick_lvid_ordinal_node ~/o/sysrap/tests/stree_load_test.sh  
    TEST=find_inst_gas ~/o/sysrap/tests/stree_load_test.sh  

    TEST=pick_lvid_ordinal_repeat_ordinal_inst_ ~/o/sysrap/tests/stree_load_test.sh 
    TEST=pick_lvid_ordinal_repeat_ordinal_inst  ~/o/sysrap/tests/stree_load_test.sh 
    TEST=get_frame ~/o/sysrap/tests/stree_load_test.sh 
    TEST=get_prim_aabb ~/o/sysrap/tests/stree_load_test.sh 

**/

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
    void get_frame_f4(int idx) const ; 
    void pick_lvid_ordinal_node() const ; 
    void find_inst_gas() const ; 
    void pick_lvid_ordinal_repeat_ordinal_inst_() const ; 
    void pick_lvid_ordinal_repeat_ordinal_inst() const ; 
    void get_frame() const ; 
    void get_prim_aabb() const ; 

    int main(); 
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

inline void stree_load_test::get_frame_f4(int idx) const
{
    sframe fr = {} ; 
    st->get_frame_f4( fr, idx );  

    std::cout 
       << "stree_load_test::get_frame_f4" 
       << " idx " << idx 
       << std::endl 
       << fr.desc()
       << std::endl 
       ;
    
}

inline void stree_load_test::pick_lvid_ordinal_node() const 
{
    std::cout
        << "stree_load_test::pick_lvid_ordinal_node"
        << "\n"
        << " (lvid, lvid_ordinal, n.desc ) "
        << "\n"
        ;

    for(int i=0 ; i < 150 ; i++)
    {
        for( int j=0 ; j < 2 ; j++)
        {
            int lvid = i ; 
            int lvid_ordinal = j ; 
            const snode* n = st->pick_lvid_ordinal_node( lvid, lvid_ordinal ); 
            std::cout 
               << "(" 
               << std::setw(4) << lvid 
               << " "
               << std::setw(4) << lvid_ordinal
               << ") "
               << ( n ? n->desc() : "-" )
               << "\n"
               ;
        }
    }
} 

inline void stree_load_test::find_inst_gas() const
{
    std::cout
        << "stree_load_test::find_inst_gas"
        << "\n"
        << " (gas_idx, gas_ordinal, inst_idx, inst_idx_slow ) "
        << "\n"
        ;

    for(int i=0 ; i < 10 ; i++)
    {
        int q_gas_idx = i ; 

        for(int j=0 ; j < 3 ;  j++)
        {
            int q_gas_ordinal = j ; 
            int inst_idx = st->find_inst_gas(q_gas_idx, q_gas_ordinal ); 
            int inst_idx_slow = st->find_inst_gas_slowly(q_gas_idx, q_gas_ordinal ); 

            std::cout 
                << "(" 
                << std::setw(4) << q_gas_idx
                << " "
                << std::setw(4) << q_gas_ordinal
                << " "
                << std::setw(7) << inst_idx
                << " "
                << std::setw(7) << inst_idx_slow
                << ") "
                << "\n"
                ;
        }
    }
}


inline void stree_load_test::pick_lvid_ordinal_repeat_ordinal_inst_() const 
{
    std::cout 
        << "stree_load_test::pick_lvid_ordinal_repeat_ordinal_inst_"
        << " ( lvid, lvid_ordinal, repeat_ordinal, inst_idx ) "
        << "\n"
        ;

    for(int i=0 ; i < 150 ; i++)
    for(int j=0 ; j < 2 ; j++)
    for(int k=0 ; k < 2 ; k++)
    {
        int lvid = i ; 
        int lvid_ordinal = j ; 
        int repeat_ordinal = k ; 
        int inst_idx = st->pick_lvid_ordinal_repeat_ordinal_inst_( lvid, lvid_ordinal, repeat_ordinal ); 

        std::cout 
            << "(" 
            << std::setw(4) << lvid
            << " "
            << std::setw(4) << lvid_ordinal
            << " "
            << std::setw(6) << repeat_ordinal
            << " "
            << std::setw(7) << inst_idx
            << ") "
            << "\n"
            ;
    }    
}

inline void stree_load_test::pick_lvid_ordinal_repeat_ordinal_inst() const 
{
    std::cout 
        << "stree_load_test::pick_lvid_ordinal_repeat_ordinal_inst"
        << " ( i, spec, inst_idx ) "
        << "\n"
        ;

    std::vector<std::string> v_spec = {{ "Hama:0:0", "Hama:0:1000", "NNVT", "NNVT:0", "NNVT:0:0", "NNVT:0:1000" }} ; 
    int num = v_spec.size(); 
    for(int i=0 ; i < num ; i++)
    {    
        const std::string& spec = v_spec[i] ; 
        int inst_idx = st->pick_lvid_ordinal_repeat_ordinal_inst( spec.c_str() );  

        std::cout 
            << "(" 
            << std::setw(4) << i
            << " "
            << std::setw(50) << spec
            << " "
            << std::setw(7) << inst_idx
            << ") "
            << "\n"
            ;
    }
}

inline void stree_load_test::get_frame() const 
{
    std::vector<std::string> v_spec = 
       {{ 
          "HamamatsuR12860sMask:0:0",
          "Hama:0:0", 
          "Hama:0:1000", 
          "NNVT", 
          "NNVT:0", 
          "NNVT:0:0", 
          "NNVT:0:1000" 
       }} ; 
    int num = v_spec.size(); 
    for(int i=0 ; i < num ; i++)
    {    
        const std::string& spec = v_spec[i] ; 
        sfr fr = st->get_frame(spec.c_str()); 
        std::cout << fr ;  
    }
}

inline void stree_load_test::get_prim_aabb() const 
{
    std::cout << "stree_load_test::get_prim_aabb \n" ; 
    std::array<double,6> pbb ; 

    for(int i=0 ; i < 3 ; i++)
    {
        int num_nd = 0 ; 
        const char* label = nullptr ; 
        switch(i)
        {
            case 0: num_nd = st->nds.size()       ; label = "st->nds"       ; break ; 
            case 1: num_nd = st->rem.size()       ; label = "st->rem"       ; break ; 
            case 2: num_nd = st->inst_nidx.size() ; label = "st->inst_nidx" ; break ; 
        }

        std::cout << label << " " << num_nd << std::endl ;   

        for(int j=0 ; j < 3 ; j++)
        {
            int k0 = 0 ;   
            int k1 = 0 ;   
            if( j == 0 )
            {
                k0 = 0 ; 
                k1 = k0 + 20 ; 
            }
            else if( j == 1 )
            {
                k0 = num_nd/2  ; 
                k1 = k0 + 20 ; 
            }
            else if( j == 2 )
            {
                k0 = num_nd - 20   ; 
                k1 = num_nd ; 
            }

            for(int k=k0 ; k < k1 ; k++)
            {
                const snode* node = nullptr ; 
                if(      i == 0 ) node = &st->nds[k] ; 
                else if( i == 1 ) node = &st->rem[k] ; 
                else if( i == 2 ) node = &st->nds[st->inst_nidx[k]] ; 
                st->get_prim_aabb( pbb.data(), *node, nullptr ); 
                std::cout 
                    << " [" << std::setw(5) << k << "]" 
                    << " : " 
                    << s_bb::Desc<double>( pbb.data() ) 
                    << std::endl 
                    ;   
            } 
        }
    }
}



inline int stree_load_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "get_frame_f4"); 

    int LVID = ssys::getenvint("LVID",  0); 
    int NDID = ssys::getenvint("NDID",  0); 
    int IIDX = ssys::getenvint("IIDX",  0); 

    if(strcmp(TEST, "get_combined_transform")==0)
    {
        get_combined_transform(LVID, NDID );  
    }
    else if(strcmp(TEST, "get_inst") == 0)
    {
        get_inst(IIDX) ; 
    }
    else if(strcmp(TEST, "get_frame_f4") == 0)
    {
        get_frame_f4(IIDX); 
    }
    else if(strcmp(TEST, "pick_lvid_ordinal_node") == 0)
    {
        pick_lvid_ordinal_node();
    }
    else if(strcmp(TEST, "find_inst_gas") == 0)
    {
        find_inst_gas();
    }
    else if(strcmp(TEST, "pick_lvid_ordinal_repeat_ordinal_inst_") == 0)
    {
        pick_lvid_ordinal_repeat_ordinal_inst_();
    }
    else if(strcmp(TEST, "pick_lvid_ordinal_repeat_ordinal_inst") == 0)
    {
        pick_lvid_ordinal_repeat_ordinal_inst();
    }
    else if(strcmp(TEST, "get_frame") == 0)
    {
        get_frame();
    }
    else if(strcmp(TEST, "get_prim_aabb") == 0)
    {
        get_prim_aabb();
    }
    

    return 0 ; 
}


int main(int argc, char** argv)
{
    stree st ; 
    int rc = st.load("$BASE"); 
    if( rc != 0 ) return rc ; 
    std::cout << st.desc() ; 

    stree_load_test test(&st); 
    return test.main();  
}
