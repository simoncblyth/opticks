#include "OPTICKS_LOG.hh"

#include "SPath.hh"
#include "SSys.hh"
#include "U4GDML.h"
#include "U4Tree.h"

const char* FOLD = SPath::Resolve("$TMP/U4TreeTest", DIRPATH); 


void test_saveload_get_children(const stree& tree0, const stree& tree1, int nidx )
{
    std::vector<int> children0 ; 
    std::vector<int> children1 ; 
    tree0.get_children(children0, nidx); 
    tree1.get_children(children1, nidx); 

    if( nidx % 10000 == 0 )
    std::cout << " nidx " << nidx << " children " << stree::Desc(children0) << std::endl ; 

    assert( stree::Compare(children0, children1) == 0 ); 
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
    assert( stree::Compare(progeny0, progeny1) == 0 ); 
    std::cout << " nidx " << nidx << " progeny " << stree::Desc(progeny0) << std::endl ; 
}

void test_saveload(const stree& st0)
{
    std::cout << "[ st0.save " << std::endl ; 
    st0.save(FOLD); 
    std::cout << "] st0.save  " << st0.desc() << std::endl ; 

    stree st1 ; 
    std::cout << "[ st1.load " << std::endl ; 
    st1.load(FOLD);  
    std::cout << "] st1.load " << st1.desc() << std::endl ; 

    test_saveload_get_children(st0, st1); 
    test_saveload_get_progeny_r(st0, st1); 
}

void test_load(const char* fold)
{
    stree st ;
    st.load(fold); 
    std::cout << "st.desc_sub" << std::endl << st.desc_sub() << std::endl ; 

    // see sysrap/tests/stree_test.cc for stree exercises 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* path = SPath::Resolve("$SomeGDMLPath", NOOP ) ; 
    G4VPhysicalVolume* world = U4GDML::Read(path) ;  

    stree st ; 
    U4Tree tree(&st, world) ;

    st.classifySubtrees(); 
    st.save(FOLD); 

    return 0 ;  
}
