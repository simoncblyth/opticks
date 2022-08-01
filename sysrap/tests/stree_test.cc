#include <cstdlib>
#include <iostream>
#include <glm/gtx/string_cast.hpp>

#include "ssys.h"
#include "stree.h"

const char* FOLD = getenv("FOLD");  

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

void test_get_transform_product(const stree& st)
{
    const char* q_soname = "HamamatsuR12860sMask_virtual" ;  

    for(int q_ordinal=0 ; q_ordinal < 10 ; q_ordinal++ )
    {
        int nidx = st.find_lvid_node(q_soname, q_ordinal ); 
        std::cout << " q_soname " << q_soname << " q_ordinal " << q_ordinal << " nidx " << nidx << std::endl ; 

        glm::tmat4x4<double> tr(1.) ; 
        st.get_transform_product(tr, nidx); 

        std::cout << "tr\n" << glm::to_string(tr) << std::endl ; 
    }
}


void test_desc_progeny(const stree& st, const char* qname )
{
    std::vector<int> nodes ; 
    st.find_lvid_nodes(nodes, qname); 
    std::cout << "st.find_lvid_nodes(nodes, qname)   qname " << qname << " nodes.size " << nodes.size() << std::endl ; 
    std::cout << "st.desc_nodes(nodes, 20)" << std::endl << st.desc_nodes(nodes, 20) << std::endl ; 

    std::cout << "st.desc_progeny(nodes[0])" << std::endl << st.desc_progeny(nodes[0]) << std::endl; 
}

int main(int argc, char** argv)
{
    stree st ; 
    st.load(FOLD); 

    std::cout << "st.desc_sub" << std::endl << st.desc_sub() << std::endl ;

    st.disqualifyContainedRepeats() ; 
    st.sortSubtrees(); 
    // TODO: these should be done at creation, not postload

    std::cout << "st.desc_sub" << std::endl << st.desc_sub() << std::endl ;

    /*
    test_get_nodes(st); 
    test_find_lvid_node(st); 
    test_get_transform_product(st); 
    */
    test_desc_progeny(st, "sWall"); 


    return 0 ; 
}
