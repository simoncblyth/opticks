#include <cstdlib>
#include <iostream>

#include "ssys.h"
#include "stree.h"

const char* FOLD = getenv("FOLD");  

int main(int argc, char** argv)
{
    stree st ; 
    st.load(FOLD); 

    std::cout << "st.desc_sub" << std::endl << st.desc_sub() << std::endl ;

    int nidx = ssys::getenvint("NIDX", 1000)  ;   
    std::vector<int> ancestors ; 
    st.get_ancestors(ancestors, nidx) ; 

    std::cout << "st.get_ancestors NIDX " << nidx << " " << stree::Desc(ancestors) << std::endl ; 
    std::cout << st.desc_nodes(ancestors) << std::endl ; 
    std::cout << st.desc_node(nidx) << std::endl  ;   

    std::cout << " st.desc_ancestry " << std::endl ; 
    std::cout << st.desc_ancestry(nidx, true) << std::endl ; 

    int sidx = ssys::getenvint("SIDX", 0);  
    unsigned edge = ssys::getenvunsigned("EDGE", 10u); 

    const char* k = st.subs_freq->get_key(sidx); 
    unsigned    v = st.subs_freq->get_freq(sidx); 

    std::vector<int> nodes ; 
    st.get_nodes(nodes, k );  

    std::cout << " sidx " << sidx << " k " << k  << " v " << v << " nodes.size" << nodes.size() << std::endl ; 
    std::cout << st.desc_nodes(nodes, edge) ;   

    return 0 ; 
}
