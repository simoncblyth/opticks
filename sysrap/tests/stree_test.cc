#include <cstdlib>
#include <iostream>
#include <glm/gtx/string_cast.hpp>

#include "ssys.h"
#include "strid.h"
#include "stree.h"

const char* STBASE = getenv("STBASE");  

void test_get_ancestors(const stree& st)
{
    int nidx = ssys::getenvint("NIDX", 1000)  ;   
    std::vector<int> ancestors ; 
    st.get_ancestors(ancestors, nidx) ; 

    std::cout << "st.get_ancestors NIDX " << nidx << " " << stree::Desc(ancestors) << std::endl ; 
    std::cout << st.desc_nodes(ancestors) << std::endl ; 
    std::cout << st.desc_node(nidx) << std::endl  ;   

    std::cout << " st.desc_ancestry " << std::endl ; 
    std::cout << st.desc_ancestry(nidx, true) << std::endl ; 
}

void test_get_nodes(const stree& st)
{
    int sidx = ssys::getenvint("SIDX", 0);  
    unsigned edge = ssys::getenvunsigned("EDGE", 10u); 

    const char* k = st.subs_freq->get_key(sidx); 
    unsigned    v = st.subs_freq->get_freq(sidx); 

    std::vector<int> nodes ; 
    st.get_nodes(nodes, k );  

    std::cout << " sidx " << sidx << " k " << k  << " v " << v << " nodes.size" << nodes.size() << std::endl ; 
    std::cout << st.desc_nodes(nodes, edge) ;   


    int nidx = nodes[0] ; 
    std::cout << st.desc_ancestry(nidx, true) << std::endl ; 

    std::vector<int> progeny ; 
    st.get_progeny(progeny, nidx ); 

    std::cout << "st.desc_nodes(progeny, 20)" << std::endl << st.desc_nodes(progeny, 20) << std::endl ; 

    const char* qname = "sWall" ;  
    int lvid = st.find_lvid(qname) ; 
    std::cout << " qname " << qname << " lvid " << lvid << std::endl ; 

    std::vector<int> lvid_nodes ; 
    st.find_lvid_nodes(lvid_nodes, lvid); 
    std::cout << "st.desc_nodes(lvid_nodes, 20)" << std::endl << st.desc_nodes(lvid_nodes, 20) << std::endl ; 

    std::vector<int> progeny_lvn0 ; 
    st.get_progeny(progeny_lvn0, lvid_nodes[0] ); 

    std::cout << "st.desc_nodes(progeny_lvn0, 1000)" << std::endl << st.desc_nodes(progeny_lvn0, 1000) << std::endl ; 
}



void test_find_lvid_node(const stree& st)
{
    const char* q_soname = "HamamatsuR12860sMask_virtual" ;  
    for(int q_ordinal=0 ; q_ordinal < 10 ; q_ordinal++ )
    {
        int nidx = st.find_lvid_node(q_soname, q_ordinal ); 
        std::cout << " q_soname " << q_soname << " q_ordinal " << q_ordinal << " nidx " << nidx << std::endl ; 

        std::string q_spec = sstr::Format_("%s:%d:%d", q_soname, 0, q_ordinal ); 
        int nidx2 = st.find_lvid_node(q_spec.c_str()); 

        std::cout << " q_spec " << q_spec << " nidx2 " << nidx2 << std::endl ; 

        assert( nidx == nidx2 ); 
    }
}

void test_get_m2w_product_get_w2m_product(const stree& st, const char* q_soname )
{
    double delta_cut = 1e-10 ; 

    std::vector<int> nodes ; 
    st.find_lvid_nodes(nodes, q_soname); 
    unsigned num = nodes.size(); 

    glm::tmat4x4<double> m2w(1.) ; 
    glm::tmat4x4<double> w2m(1.) ; 
    double delta_max = 0. ; 
    unsigned count = 0 ; 

    for(unsigned i=0 ; i < num ; i++ )
    {
        int nidx = nodes[i] ; 
        st.get_m2w_product(m2w, nidx, false ); 
        st.get_w2m_product(w2m, nidx, true  ); 

        glm::tmat4x4<double> m2w_w2m = m2w * w2m ; 
        glm::tmat4x4<double> w2m_m2w = w2m * m2w ; 

        double delta0 = strid::DiffFromIdentity(m2w_w2m); 
        double delta1 = strid::DiffFromIdentity(w2m_m2w); 
        double delta = std::max(delta0, delta1); 
        if( delta > delta_max ) delta_max = delta ; 

        if( delta > delta_cut )
        {
            count += 1 ; 
            std::cout << " i " << i << " nidx " << nidx << " delta " << std::scientific << delta << std::endl ; 
            std::cout << strid::Desc_("m2w", "w2m", "m2w_w2m", m2w, w2m, m2w_w2m ) ; 
        }
    }
    std::cout 
        << " soname " << std::setw(50) << q_soname 
        << " num " << std::setw(7) << num
        << " delta_max " << std::setw(10) << std::scientific << delta_max 
        << " delta_cut " << std::setw(10) << std::scientific << delta_cut 
        << " count " << std::setw(8) << count
        << std::endl 
        ; 
}

void test_get_m2w_product_get_w2m_product(const stree& st)
{
    std::vector<std::string> sonames ; 
    st.get_sub_sonames(sonames) ; 
    std::cout << "[ test_get_m2w_product_get_w2m_product sonames.size " << sonames.size() << std::endl ; 
    std::cout << "testing that m2w and w2m product transforms for all instances are inverses of each other " << std::endl ; 

    for(unsigned i=0 ; i < sonames.size() ; i++)
    {
        const char* soname = sonames[i].c_str(); 
        test_get_m2w_product_get_w2m_product(st, soname ) ;  
    }
    std::cout << "] test_get_m2w_product_get_w2m_product sonames.size " << sonames.size() << std::endl ; 
}

void test_m2w_w2m(const stree& st)
{
    std::cout << "test_m2w_w2m : check that all m2w w2m pairs are effective inverses of each other " << std::endl ; 
    std::cout << "st.m2w.size " << st.m2w.size() << std::endl ; 
    std::cout << "st.w2m.size " << st.w2m.size() << std::endl ; 
    assert( st.m2w.size() == st.w2m.size() ); 
    unsigned num = st.m2w.size();

    unsigned i0 = 0 ; 
    unsigned i1 = num ;

    double delta_cut = 1e-11 ;  
    unsigned count = 0 ; 
    unsigned max_delta_idx = 0 ; 
    double max_delta = 0. ; 

    for(unsigned i=i0 ; i < i1 ; i++)
    {
        assert( i < num ); 
  
        const glm::tmat4x4<double>& m2w = st.m2w[i] ; 
        const glm::tmat4x4<double>& w2m = st.w2m[i] ; 
        const glm::tmat4x4<double>  m2w_w2m = m2w*w2m ; 
        const glm::tmat4x4<double>  w2m_m2w = w2m*m2w ; 

        double delta_0 = strid::DiffFromIdentity(m2w_w2m) ; 
        double delta_1 = strid::DiffFromIdentity(w2m_m2w) ; 
        double delta   = std::max( delta_0, delta_1 ); 

        if( delta > max_delta ) 
        {
            max_delta = delta ;
            max_delta_idx = i ; 
        }  

        if( delta > delta_cut )
        {
            count += 1 ; 
            std::cout 
                << " i " << std::setw(7) << i 
                << " delta_0 " << std::setw(10) << std::scientific << delta_0 
                << " delta_1 " << std::setw(10) << std::scientific << delta_1
                << " delta "   << std::setw(10) << std::scientific << delta
                << " delta_cut "   << std::setw(10) << std::scientific << delta_cut
                << std::endl
                ; 

            std::cout << strid::Desc_("m2w","w2m","m2w_w2m", m2w, w2m, m2w_w2m ) << std::endl ; 
            std::cout << strid::Desc_("m2w","w2m","w2m_m2w", m2w, w2m, w2m_m2w ) << std::endl ; 
        }
    }
    std::cout 
        << " i0 " << i0 
        << " i1 " << i1 
        << " delta_cut " << std::scientific << delta_cut 
        << " max_delta " << std::scientific << max_delta 
        << " count " << count 
        << " max_delta_idx " << max_delta_idx 
        << std::endl
        ; 
}


void test_desc_progeny(const stree& st, const char* qname )
{
    std::vector<int> nodes ; 
    st.find_lvid_nodes(nodes, qname); 
    std::cout << "st.find_lvid_nodes(nodes, qname)   qname " << qname << " nodes.size " << nodes.size() << std::endl ; 
    std::cout << "st.desc_nodes(nodes, 20)" << std::endl << st.desc_nodes(nodes, 20) << std::endl ; 
    std::cout << "st.desc_progeny(nodes[0])" << std::endl << st.desc_progeny(nodes[0]) << std::endl; 
}


void test_get_factor_nodes(const stree& st)
{
    unsigned num_factor = st.get_num_factor(); 
    std::cout << "test_get_factor_nodes num_factor " << num_factor << std::endl ; 

    for(unsigned i=0 ; i < num_factor ; i++)
    {
        std::vector<int> nodes ; 
        st.get_factor_nodes(nodes, i); 
        std::cout << std::setw(3) << i << " nodes " << nodes.size() << std::endl ;  
    }

    std::cout << st.desc_factor() << std::endl ; 
}

void test_traverse(const stree& st)
{
    st.traverse(); 
}
void test_reorderSensors(stree& st)
{
    st.reorderSensors(); 
    st.clear_inst(); 
    st.add_inst(); 

    st.save(STBASE, "stree_reorderSensors" );  
}

void test_get_sensor_id(const stree& st)
{
    std::vector<int> sensor_id ; 
    st.get_sensor_id(sensor_id); 
    unsigned num_sensor = sensor_id.size() ; 
    std::cout << "test_get_sensor_id  num_sensor " << num_sensor << std::endl ; 

    unsigned edge = 10 ; 

    int pid = -1 ; 
    int offset = 0 ; 

    for(unsigned i=0 ; i < num_sensor  ; i++)
    {
        int sid = sensor_id[i] ; 
        int nid = i < num_sensor - 1 ? sensor_id[i+1] : sid ; 

        bool head = i < edge ; 
        bool tail = i > (num_sensor - edge) ; 
        bool tran = std::abs(sid - pid) > 1 || std::abs( sid - nid ) > 1  ; 

        if(tran) offset = 0 ; 
        offset += 1 ;

        bool post_tran = offset < 5 ; 
 

        if( head || tail || tran || post_tran  ) 
        {
            std::cout 
                << " i " << std::setw(7) << i 
                << " sensor_id " << std::setw(7) << sid 
                << std::endl ; 
         }
         else if ( i == edge ) 
         {
            std::cout  << "..." << std::endl ; 
         }
         pid = sid ; 

    }
}

void test_desc_m2w_product(const stree& st)
{
    int ins_idx = ssys::getenvint("INS_IDX", 1 ); 
    int num_inst = int(st.inst_nidx.size()) ; 
    if(ins_idx < 0 ) ins_idx += num_inst ; 
    assert( ins_idx < num_inst ); 

    int nidx = st.inst_nidx[ins_idx] ;  
    std::cout 
         << "st.inst_nidx.size " << num_inst 
         << " ins_idx INS_IDX " << ins_idx  
         << " nidx " << nidx 
         << std::endl 
         ; 

    bool reverse = false ; 
    std::cout << st.desc_m2w_product(nidx, reverse) << std::endl ;  
} 

int main(int argc, char** argv)
{
    stree st ; 
    st.load(STBASE); 

    std::cout << "st.desc_sub(false)" << std::endl << st.desc_sub(false) << std::endl ;
    //std::cout << "st.desc_sub(true)"  << std::endl << st.desc_sub(true)  << std::endl ;

    /*
    test_m2w_w2m(st); 
    test_get_m2w_product_get_w2m_product(st) ;  
    test_get_nodes(st); 
    test_find_lvid_node(st); 
    test_desc_progeny(st, "sWall"); 
    test_get_factor_nodes(st); 
    test_traverse(st); 
    test_reorderSensors(st); 
    test_get_sensor_id(st); 
    test_desc_m2w_product(st); 
    */

    test_get_ancestors(st); 


    return 0 ; 
}
