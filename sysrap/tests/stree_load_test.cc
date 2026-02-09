/**
stree_load_test.cc
===================

::

    ~/o/sysrap/tests/stree_load_test.sh

    TEST=get_inst IIDX=100                      ~/o/sysrap/tests/stree_load_test.sh
    TEST=pick_lvid_ordinal_node                 ~/o/sysrap/tests/stree_load_test.sh
    TEST=find_inst_gas                          ~/o/sysrap/tests/stree_load_test.sh

    TEST=pick_lvid_ordinal_repeat_ordinal_inst_ ~/o/sysrap/tests/stree_load_test.sh
    TEST=pick_lvid_ordinal_repeat_ordinal_inst  ~/o/sysrap/tests/stree_load_test.sh
    TEST=get_frame                              ~/o/sysrap/tests/stree_load_test.sh
    TEST=get_prim_aabb                          ~/o/sysrap/tests/stree_load_test.sh

    TEST=desc_factor_nodes FIDX=0               ~/o/sysrap/tests/stree_load_test.sh
    TEST=desc_repeat_node RIDX=0 RORD=0         ~/o/sysrap/tests/stree_load_test.sh

**/

#include "ssys.h"
#include "stree.h"
#include "stra.h"
#include "strid.h"
#include "sn.h"
#include "SBitSet.h"


struct stree_load_test
{
    const char* TEST ;
    const stree* st ;

    stree_load_test();
    void init();

    int get_combined_transform(int LVID, int NDID  );
    int get_inst(int idx) const ;
    int get_frame_f4(int idx) const ;
    int pick_lvid_ordinal_node() const ;
    int find_inst_gas() const ;
    int pick_lvid_ordinal_repeat_ordinal_inst_() const ;
    int pick_lvid_ordinal_repeat_ordinal_inst() const ;
    int get_frame() const ;
    int get_frame_scan_(const char* solid, int i0, int i1, int j0, int j1) const ;
    int get_frame_scan() const ;
    int get_frame_MOI() const ;
    int get_prim_aabb() const ;
    int desc_repeat_index() const ;

    int get_global_aabb() const ;
    int get_global_aabb_check() const ;

    int get_global_aabb_sibling_overlaps() const ;


    int desc_factor_nodes(int fidx) const ;
    int desc_repeat_node(int ridx, int rord) const ;
    int desc_repeat_nodes() const ;

    int desc_nds() const ;
    int desc_rem() const ;
    int desc_tri() const ;
    int desc_NRT() const ;

    int desc_node_ELVID() const ;
    int desc_node_ECOPYNO() const ;
    int desc_node_EBOUNDARY() const ;

    int desc_node_solids() const ;
    int desc_solids() const ;
    int desc_solid(int lvid) const ;
    int desc() const ;
    int save_desc(const char* fold) const ;
    int make_tree_digest() const ;

    int desc_nodes_with_center_within_ce() const ;


    int main();
};

inline stree_load_test::stree_load_test()
    :
    TEST(ssys::getenvvar("TEST", nullptr)),
    st(nullptr)
{
    init();
}

inline void stree_load_test::init()
{
    bool noload = strcmp(TEST,"get_global_aabb_check") == 0 ;
    if(!noload)
    {
        st = stree::Load();
        if( st == nullptr ) std::cout << "stree_load_test::init FAILED TO LOAD TREE \n" ;
    }

    std::cout << "[stree_load_test::init\n" ;
    std::cout << ( st ? st->desc_id() : "-" ) << "\n" ;
    std::cout << "]stree_load_test::init\n" ;
}


inline int stree_load_test::get_combined_transform( int LVID, int NDID )
{
    std::cout
        << "stree_load_test::get_combined_transform"
        << " LVID " << LVID
        << " NDID " << NDID
        << std::endl
        ;

    std::vector<snode> nodes ;  // structural volume nodes with the LVID, could be thousands
    st->find_lvid_nodes_(nodes, LVID, 'N') ;

    int num_nodes = nodes.size();
    std::cout << " VOL " << st->desc_nodes_(nodes) ;


    std::vector<sn*> nds ;    // CSG constituent nodes of the LV
    sn::GetLVNodes(nds, LVID );
    int num_nds = nds.size();
    std::cout << " sn::Desc(nds) " << std::endl << sn::Desc(nds) ;

    assert( num_nodes > 0 && num_nds > 1 );

    std::vector<glm::tmat4x4<double>> tvs ;
    tvs.reserve(num_nodes*2);

    for(int i=0 ; i < num_nodes ; i++)
    {
        bool dump_NDID = i == NDID ;

        const snode& node = nodes[i] ;
        const sn*  nd = nds[0] ;

        glm::tmat4x4<double> t(1.) ;
        glm::tmat4x4<double> v(1.) ;

        std::stringstream* out = dump_NDID ? new std::stringstream : nullptr ;
        stree::VTR* t_stack = nullptr ;

        st->get_combined_transform(t, v, node, nd, out, t_stack );

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
    return 0 ;
}

inline int stree_load_test::get_inst(int idx) const
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
    return 0 ;
}

inline int stree_load_test::get_frame_f4(int idx) const
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

    return 0 ;
}

inline int stree_load_test::pick_lvid_ordinal_node() const
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
            const snode* n = st->pick_lvid_ordinal_node( lvid, lvid_ordinal, 'N' );
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
    return 0 ;
}

inline int stree_load_test::find_inst_gas() const
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
    return 0 ;
}


inline int stree_load_test::pick_lvid_ordinal_repeat_ordinal_inst_() const
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
    return 0 ;
}

inline int stree_load_test::pick_lvid_ordinal_repeat_ordinal_inst() const
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
    return 0 ;
}

inline int stree_load_test::get_frame() const
{
    std::vector<std::string> v_spec =
       {{
          "HamamatsuR12860sMask:0:0",
          "Hama:0:0",
          "Hama:0:1000",
          "NNVT",
          "NNVT:0",
          "NNVT:0:0",
          "NNVT:0:1000",
          "sDeadWater:0:-1",
          "GZ1.A06_07_FlangeI_Web_FlangeII:0:-1",
          "GZ1.B06_07_FlangeI_Web_FlangeII:0:-1",
          "GZ1.A06_07_FlangeI_Web_FlangeII:15:-1",
          "GZ1.B06_07_FlangeI_Web_FlangeII:15:-1"
       }} ;
    int num = v_spec.size();
    for(int i=0 ; i < num ; i++)
    {
        const std::string& spec = v_spec[i] ;
        sfr fr = st->get_frame(spec.c_str());
        std::cout << fr ;
    }
    return 0 ;
}

inline int stree_load_test::get_frame_MOI() const
{
    const char* MOI = ssys::getenvvar("MOI", nullptr);
    if(!MOI) return 1 ;

    sfr mfr = st->get_frame(MOI);
    std::cout << "stree_load_test::get_frame_MOI\n" <<  MOI << "\n" << mfr << "\n"  ;
    return 0 ;
}


inline int stree_load_test::get_frame_scan_(const char* solid, int i0, int i1, int j0, int j1) const
{
    const char* fmt = "%s:%d:%d" ;
    std::cout
        << "stree_load_test::get_frame_scan_"
        << " fmt [" << fmt << "] "
        << " solid [" << ( solid ? solid : "-" ) << "] "
        << " i0 " << i0 << " i1 " << i1
        << " j0 " << j0 << " j1 " << j1
        << "\n"
        ;

    for(int i=i0 ; i < i1 ; i++)
    for(int j=j0 ; j <= j1 ; j++)
    {
        std::string _moi = sstr::Format_(fmt, solid, i, j);
        const char* moi = _moi.c_str();
        bool has_frame = st->has_frame(moi);
        std::cout << "\n\n" << moi << " has_frame " <<  ( has_frame ? "YES" : "NO" ) << "\n" ;

        if(has_frame)
        {
            sfr mfr = st->get_frame(moi);
            //std::cout <<  mfr << "\n" ;
            std::cout <<  mfr.desc_ce() << "\n" ;
        }
    }
    return 0 ;
}
inline int stree_load_test::get_frame_scan() const
{
    const char* solid = "solidXJfixture" ;
    get_frame_scan_(solid, 0, 60, -1, -1 );
    return 0 ;
}


inline int stree_load_test::get_prim_aabb() const
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

                std::ostream* out = nullptr ;
                stree::VTR* t_stack = nullptr ;
                st->get_prim_aabb( pbb.data(), *node, out, t_stack );
                std::cout
                    << " [" << std::setw(5) << k << "]"
                    << " : "
                    << s_bb::Desc<double>( pbb.data() )
                    << std::endl
                    ;
            }
        }
    }
    return 0 ;
}


/**
stree_load_test::desc_repeat_index
-------------------------------------


::

    stree_load_test::desc_repeat_index
    [stree::desc_repeat_index
    [stree::desc_repeat_index N[nds.npy]
       0 :       3431
       1 :     128000
       2 :     113913
       3 :      59268
       4 :       9596
       5 :       2400
       6 :        590
       7 :        590
       8 :       1770           ## 590*3 = 1770
       9 :      64480           ## 496*130 =
      10 :        370
      11 :       2118           ## 353*6 = 2118
    TOT: :     386526
    ]stree::desc_repeat_index N[nds.npy]
    [stree::desc_repeat_index R[rem.npy]
       0 :       3087
    TOT: :       3087
    ]stree::desc_repeat_index R[rem.npy]
    [stree::desc_repeat_index T[tri.npy]
       0 :        344
    TOT: :        344
    ]stree::desc_repeat_index T[tri.npy]
     N_tot     386526 N0       3431 (R_tot + T_tot)       3431 N0_expect YES
     R_tot       3087 R0       3087 R0_expect YES
     T_tot        344 T0        344 T0_expect YES
    ]stree::desc_repeat_index



    (ok) A[blyth@localhost CSGFoundry]$ cat.py mmlabel.txt
    0    3087:sWorld
    1    5:PMT_3inch_pmt_solid
    2    9:NNVTMCPPMTsMask_virtual
    3    12:HamamatsuR12860sMask_virtual
    4    4:mask_PMT_20inch_vetosMask_virtual
    5    4:PMT_8inch_pmt_solid
    6    1:sStrutBallhead
    7    1:base_steel
    8    3:uni_acrylic1
    9    130:sPanel
    10   1:sStrut_0
    11   6:PMT_20inch_pmt_solid_head
    12   344:ConnectingCutTube_0

    THE SOLID WITH INDEX 12 IS AN EXTRA FOR THE TRIANGULATED NODES
    THAT ALL HAVE repeat_index zero

**/


inline int stree_load_test::desc_repeat_index() const
{
    std::cout << "[stree_load_test::desc_repeat_index\n" ;
    std::cout << st->desc_repeat_index() << "\n" ;
    std::cout << "]stree_load_test::desc_repeat_index\n" ;
    return 0 ;
}


inline int stree_load_test::get_global_aabb() const
{
    std::cout << "[stree_load_test::get_global_aabb \n" ;

    NPFold* f = st->get_global_aabb();
    f->save("$TMPFOLD/get_global_aabb");

    std::cout << "]stree_load_test::get_global_aabb \n" ;
    return 0 ;
}


/**
stree_load_test::get_global_aabb_check
----------------------------------------

This approach yields too many overlaps to wade thru

**/

inline int stree_load_test::get_global_aabb_check() const
{
    NPFold* f = NPFold::Load("$TMPFOLD/get_global_aabb");
    const NP* bb = f->get("bb");
    const NP* ii = f->get("ii");

    const double* bbv = bb->cvalues<double>();
    const size_t* iiv = ii->cvalues<size_t>();

    std::cout << "[stree_load_test::get_global_aabb_check \n" ;
    std::cout << " bb " << ( bb ? bb->sstr() : "-" ) << "\n";
    std::cout << " ii " << ( ii ? ii->sstr() : "-" ) << "\n";

    enum { bb_nj = 6 };

    assert( bb->shape[0] == ii->shape[0] );
    assert( bb->shape[1] == bb_nj );
    assert( ii->shape[1] == snode::NV );

    struct Overlap
    {
        size_t i, j ;
    };

    std::vector<Overlap> overlap ;

    size_t num_bb = bb->shape[0] ;
    for( size_t i = 0 ; i < num_bb ; i++ )
    for( size_t j = 0 ; j < num_bb ; j++ )
    {
        if( j >= i ) continue ;
        const double* a = &bbv[bb_nj*i+0] ;
        const double* b = &bbv[bb_nj*j+0] ;
        bool has_overlap = s_bb::HasOverlap<double>( a, b );
        if(has_overlap) overlap.push_back( {i, j} );
    }
    std::cout
       << " overlap " << overlap.size()
       << "\n"
       ;

    std::cout << "]stree_load_test::get_global_aabb_check \n" ;
    return 0 ;
}




inline int stree_load_test::get_global_aabb_sibling_overlaps() const
{
    std::cout << "[stree_load_test::get_global_aabb_sibling_overlaps \n" ;

    NPFold* f = st->get_global_aabb_sibling_overlaps();
    f->save("$TMPFOLD/get_global_aabb_sibling_overlaps");

    std::cout << "]stree_load_test::get_global_aabb_sibling_overlaps \n" ;
    return 0 ;
}







inline int stree_load_test::desc_factor_nodes(int fidx) const
{
    std::cout << st->desc_factor_nodes(fidx) << "\n" ;
    return 0 ;
}


inline int stree_load_test::desc_repeat_node(int ridx, int rord) const
{
    std::cout << st->desc_repeat_node(ridx, rord) << "\n" ;
    return 0 ;
}
inline int stree_load_test::desc_repeat_nodes() const
{
    std::cout << st->desc_repeat_nodes() << "\n" ;
    return 0 ;
}


inline int stree_load_test::desc_nds() const
{
    std::cout << st->desc_nds() << "\n" ;
    return 0 ;
}
inline int stree_load_test::desc_rem() const
{
    std::cout << st->desc_rem() << "\n" ;
    return 0 ;
}
inline int stree_load_test::desc_tri() const
{
    std::cout << st->desc_tri() << "\n" ;
    return 0 ;
}
inline int stree_load_test::desc_NRT() const
{
    std::cout << st->desc_NRT() << "\n" ;
    return 0 ;
}

inline int stree_load_test::desc_node_ELVID() const
{
    std::cout << st->desc_node_ELVID() << "\n" ;
    return 0 ;
}

inline int stree_load_test::desc_node_ECOPYNO() const
{
    std::cout << st->desc_node_ECOPYNO() << "\n" ;
    return 0 ;
}
inline int stree_load_test::desc_node_EBOUNDARY() const
{
    std::cout << st->desc_node_EBOUNDARY() << "\n" ;
    return 0 ;
}




inline int stree_load_test::desc_node_solids() const
{
    std::cout << st->desc_node_solids() ;
    return 0 ;
}

/**
stree_load_test::desc_solids
-----------------------------

solids vector not persisted
**/

inline int stree_load_test::desc_solids() const
{
    std::cout << st->desc_solids() ;
    return 0 ;
}
inline int stree_load_test::desc_solid(int lvid) const
{
    std::cout << st->desc_solid(lvid) ;

    std::vector<int> nodes ;
    char src = 'N' ;
    st->find_lvid_nodes(nodes, lvid, src );

    std::cout
        << "st.find_lvid_nodes (structural nodes) "
        << " lvid " << lvid
        << " src " << src
        << " nodes.size " << nodes.size()
        << "\n"
        ;

    for(size_t i=0 ; i < nodes.size() ; i++ )
    {
        int nidx = nodes[i];
        std::cout << " nidx " << nidx << "\n" ;

        const glm::tmat4x4<double>& m2w = st->m2w[nidx] ;
        std::cout << "m2w\n" << stra<double>::Desc(m2w) << "\n" ;

        const glm::tmat4x4<double>& gtd = st->gtd[nidx] ;
        std::cout << "gtd\n" << stra<double>::Desc(m2w) << "\n" ;
    }




    std::vector<sn*> nds ;    // CSG constituent nodes of the LV
    sn::GetLVNodes(nds, lvid );
    int num_nds = nds.size();
    std::cout
        << " sn::Desc(nds) (csg nodes) lvid " << lvid << " num_nds " << num_nds
        << "\n"
        << sn::Desc(nds)
        << "\n"
        << " sn::DescXform(nds) (csg nodes) lvid " << lvid << " num_nds " << num_nds
        << sn::DescXform(nds)
        << "\n"
        << " sn::DescXformFull(nds) (csg nodes) lvid " << lvid << " num_nds " << num_nds
        << sn::DescXformFull(nds)
        << "\n"
        ;

    return 0 ;
}
inline int stree_load_test::desc() const
{
    std::cout << st->desc() << "\n" ;
    return 0 ;
}
inline int stree_load_test::save_desc(const char* fold) const
{
    std::cout
        << "stree_load_test::save_desc"
        << " fold " << ( fold ? fold : "-" )
        << "\n"
        ;
    if(fold==nullptr) return 0;

    st->save_desc(fold);
    return 0 ;
}
inline int stree_load_test::make_tree_digest() const
{
    const char* tree_digest = st->get_tree_digest();

    std::vector<unsigned char> extra ;
    std::string dynamic_0 = st->make_tree_digest(extra);

    uint32_t value = 0xdeadbeef ;
    SBitSet* bs0 = SBitSet::Create<uint32_t>( value );
    bs0->serialize(extra);

    std::string dynamic_1 = st->make_tree_digest(extra);

    std::cout
        << "stree_load_test::make_tree_digest"
        << "\n"
        << " tree_digest " << ( tree_digest ? tree_digest : "-" )
        << "\n"
        << " dynamic_0   " << dynamic_0
        << "\n"
        << " dynamic_1   " << dynamic_1
        << "\n"
        ;

    return 0 ;
}

inline int stree_load_test::desc_nodes_with_center_within_ce() const
{
     std::vector<double>* ce = ssys::getenv_vec<double>("CE", "-6000,0,20000,1000");
     assert(ce);
     std::cout << st->desc_nodes_with_center_within_ce( ce->data() );
     return 0 ;
}


inline int stree_load_test::main()
{
    const char* TMPFOLD = ssys::getenvvar("TMPFOLD", nullptr);

    int LVID = ssys::getenvint("LVID",  0);
    int NDID = ssys::getenvint("NDID",  0);
    int IIDX = ssys::getenvint("IIDX",  0);
    int FIDX = ssys::getenvint("FIDX",  0);
    int RIDX = ssys::getenvint("RIDX",  0);
    int RORD = ssys::getenvint("RORD",  0);

    int rc = 0 ;
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    if(ALL||strcmp(TEST, "get_combined_transform")==0)   rc += get_combined_transform(LVID, NDID );
    if(ALL||strcmp(TEST, "get_inst") == 0)               rc += get_inst(IIDX) ;
    if(ALL||strcmp(TEST, "get_frame_f4") == 0)           rc += get_frame_f4(IIDX);
    if(ALL||strcmp(TEST, "pick_lvid_ordinal_node") == 0) rc += pick_lvid_ordinal_node();
    if(ALL||strcmp(TEST, "find_inst_gas") == 0)          rc += find_inst_gas();
    if(ALL||strcmp(TEST, "pick_lvid_ordinal_repeat_ordinal_inst_") == 0) rc += pick_lvid_ordinal_repeat_ordinal_inst_();
    if(ALL||strcmp(TEST, "pick_lvid_ordinal_repeat_ordinal_inst") == 0)  rc += pick_lvid_ordinal_repeat_ordinal_inst();
    if(ALL||strcmp(TEST, "get_frame") == 0)                              rc += get_frame();
    if(ALL||strcmp(TEST, "get_frame_MOI") == 0)                          rc += get_frame_MOI();
    if(ALL||strcmp(TEST, "get_frame_scan") == 0)                         rc += get_frame_scan();
    if(ALL||strcmp(TEST, "get_prim_aabb") == 0)                          rc += get_prim_aabb();

    if(ALL||strcmp(TEST, "get_global_aabb") == 0)                        rc += get_global_aabb();
    if(ALL||strcmp(TEST, "get_global_aabb_check") == 0)                  rc += get_global_aabb_check();
    if(ALL||strcmp(TEST, "get_global_aabb_sibling_overlaps") == 0)       rc += get_global_aabb_sibling_overlaps();

    if(ALL||strcmp(TEST, "desc_repeat_index") == 0)                      rc += desc_repeat_index();
    if(ALL||strcmp(TEST, "desc_factor_nodes") == 0)                      rc += desc_factor_nodes(FIDX);
    if(ALL||strcmp(TEST, "desc_repeat_node") == 0)                       rc += desc_repeat_node(RIDX, RORD);
    if(ALL||strcmp(TEST, "desc_repeat_nodes") == 0)                      rc += desc_repeat_nodes();
    if(ALL||strcmp(TEST, "desc_nds") == 0)                               rc += desc_nds();
    if(ALL||strcmp(TEST, "desc_rem") == 0)                               rc += desc_rem();
    if(ALL||strcmp(TEST, "desc_tri") == 0)                               rc += desc_tri();
    if(ALL||strcmp(TEST, "desc_NRT") == 0)                               rc += desc_NRT();
    if(ALL||strcmp(TEST, "desc_node_ELVID") == 0)                        rc += desc_node_ELVID();
    if(ALL||strcmp(TEST, "desc_node_ECOPYNO") == 0)                      rc += desc_node_ECOPYNO();
    if(ALL||strcmp(TEST, "desc_node_EBOUNDARY") == 0)                    rc += desc_node_EBOUNDARY();
    if(ALL||strcmp(TEST, "desc_node_solids") == 0)                       rc += desc_node_solids() ;
    if(ALL||strcmp(TEST, "desc_solids") == 0)                            rc += desc_solids() ;
    if(ALL||strcmp(TEST, "desc_solid") == 0)                             rc += desc_solid(LVID) ;
    if(ALL||strcmp(TEST, "desc") == 0)                                   rc += desc();
    if(ALL||strcmp(TEST, "save_desc") == 0)                              rc += save_desc(TMPFOLD);
    if(ALL||strcmp(TEST, "make_tree_digest") == 0)                       rc += make_tree_digest();
    if(ALL||strcmp(TEST, "desc_nodes_with_center_within_ce") == 0)       rc += desc_nodes_with_center_within_ce();

    return rc ;
}


int main(int argc, char** argv)
{
    stree_load_test t;
    return t.main();
}

/**
    TEST=desc_solid LVID=43 ~/o/sysrap/tests/stree_load_test.sh  run
    TEST=desc_solid LVID=124 ~/o/sysrap/tests/stree_load_test.sh  run
    TEST=desc_solid LVID=32 ~/o/sysrap/tests/stree_load_test.sh


    TEST=desc_node_ELVID ELVID=43,44,45,46 ~/o/sysrap/tests/stree_load_test.sh
    TEST=desc_node_ECOPYNO ECOPYNO=52400   ~/o/sysrap/tests/stree_load_test.sh
    TEST=desc_node_EBOUNDARY EBOUNDARY=303   ~/o/sysrap/tests/stree_load_test.sh
**/

