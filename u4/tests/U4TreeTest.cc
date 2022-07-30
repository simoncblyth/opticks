#include "OPTICKS_LOG.hh"

#include "SPath.hh"
#include "SSys.hh"
#include "U4GDML.h"
#include "U4Tree.h"

int compare( const std::vector<int>& a, const std::vector<int>& b )
{
    if( a.size() != b.size() ) return -1 ;  
    int mismatch = 0 ; 
    for(unsigned i=0 ; i < a.size() ; i++) if(a[i] != b[i]) mismatch += 1 ; 
    return mismatch ;  
}

std::string Desc(const std::vector<int>& a, unsigned edgeitems=10 )
{
    std::stringstream ss ; 
    ss << "Desc " << a.size() << " : " ; 
    for(unsigned i=0 ; i < a.size() ; i++)
    {
        if(i < edgeitems || i > (a.size() - edgeitems) ) ss << a[i] << " " ; 
        else if( i == edgeitems ) ss << "... " ; 
    }
    std::string s = ss.str(); 
    return s ; 
}

void test_saveload_get_children(const stree& tree0, const stree& tree1, int nidx )
{
    std::vector<int> children0 ; 
    std::vector<int> children1 ; 
    tree0.get_children(children0, nidx); 
    tree1.get_children(children1, nidx); 

    if( nidx % 10000 == 0 )
    std::cout << " nidx " << nidx << " children " << Desc(children0) << std::endl ; 

    assert( compare(children0, children1) == 0 ); 
}
void test_saveload_get_children(const stree& tree0, const stree& tree1)
{
    std::cout << "[ test_saveload_get_children " << std::endl ; 
    assert( tree0.nds.size() == tree1.nds.size() ); 
    for(int nidx=0 ; nidx < int(tree0.nds.size()) ; nidx++)
        test_saveload_get_children(tree0, tree1, nidx ) ; 
    std::cout << "] test_saveload_get_children " << std::endl ; 
}

void test_saveload_get_progeny_r(const stree& tree0, const stree& tree1)
{
    std::cout << "test_saveload_get_progeny_r " << std::endl ; 
    int nidx = 0 ; 
    std::vector<int> progeny0 ; 
    std::vector<int> progeny1 ; 
    tree0.get_progeny_r(progeny0, nidx); 
    tree1.get_progeny_r(progeny1, nidx); 
    assert( compare(progeny0, progeny1) == 0 ); 
    std::cout << " nidx " << nidx << " progeny " << Desc(progeny0) << std::endl ; 
}

void test_saveload(const stree& st0, const char* fold)
{
    std::cout << "[ st0.save " << std::endl ; 
    st0.save(fold); 
    std::cout << "] st0.save  " << st0.desc() << std::endl ; 

    stree st1 ; 
    std::cout << "[ st1.load " << std::endl ; 
    st1.load(fold);  
    std::cout << "] st1.load " << st1.desc() << std::endl ; 

    test_saveload_get_children(st0, st1); 
    test_saveload_get_progeny_r(st0, st1); 
}

void test_create(const char* fold)
{
    const char* path = SPath::Resolve("$SomeGDMLPath", FILEPATH) ; 
    G4VPhysicalVolume* world = U4GDML::Read(path) ;  

    stree st ; 
    U4Tree tree0(&st, world) ;
    st.classifySubtrees(); 
    st.save(fold); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* fold = SPath::Resolve("$TMP/U4TreeTest", DIRPATH); 
   // test_create(fold); 

    stree st ;
    st.load(fold); 
 
    std::cout << "st.desc" << std::endl << st.desc() << std::endl ; 

    int nidx = SSys::getenvint("NIDX", 1000)  ; 
    std::vector<int> ancestors ; 
    st.get_ancestors(ancestors, nidx) ; 

    std::cout << " get_ancestors nidx " << nidx << " " << Desc(ancestors) << std::endl ; 
    std::cout << st.desc_nodes(ancestors) << std::endl ; 
    std::cout << st.desc_node(nidx) << std::endl  ; 

    int sidx = SSys::getenvint("SIDX", 0);  
    unsigned edge = SSys::getenvunsigned("EDGE", 10u); 

    const char* k = st.subs_freq->get_key(sidx); 
    unsigned    v = st.subs_freq->get_freq(sidx); 

    std::vector<int> nodes ; 
    st.get_nodes(nodes, k ); 

    std::cout << " sidx " << sidx << " k " << k  << " v " << v << " nodes.size" << nodes.size() << std::endl ; 
    std::cout << st.desc_nodes(nodes, edge) ;   

    //test_saveload(st0, fold);  

    return 0 ;  
}
