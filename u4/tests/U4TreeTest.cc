#include "OPTICKS_LOG.hh"

#include "SPath.hh"
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

void test_get_children(const U4Tree& tree0, const U4Tree& tree1, int nidx )
{
    std::vector<int> children0 ; 
    std::vector<int> children1 ; 
    tree0.get_children(children0, nidx); 
    tree1.get_children(children1, nidx); 
    std::cout << " nidx " << nidx << " children " << Desc(children0) << std::endl ; 
    assert( compare(children0, children1) == 0 ); 
}
void test_get_children(const U4Tree& tree0, const U4Tree& tree1)
{
    assert( tree0.nds.size() == tree1.nds.size() ); 
    for(int nidx=0 ; nidx < int(tree0.nds.size()) ; nidx++)
        test_get_children(tree0, tree1, nidx ) ; 
}

void test_get_progeny_r(const U4Tree& tree0, const U4Tree& tree1)
{
    int nidx = 0 ; 
    std::vector<int> progeny0 ; 
    std::vector<int> progeny1 ; 
    tree0.get_progeny_r(progeny0, nidx); 
    tree1.get_progeny_r(progeny1, nidx); 
    assert( compare(progeny0, progeny1) == 0 ); 
    std::cout << " nidx " << nidx << " progeny " << Desc(progeny0) << std::endl ; 
}






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* path = SPath::Resolve("$SomeGDMLPath", FILEPATH) ; 
    const char* fold = SPath::Resolve("$TMP/U4TreeTest", DIRPATH); 

    G4VPhysicalVolume* world = U4GDML::Read(path) ;  

    U4Tree tree0(world) ;
    tree0.save(fold); 
    std::cout << "tree0  " << tree0.desc() << std::endl ; 

    U4Tree tree1 ; 
    tree1.load(fold);  
    std::cout << "tree1 " << tree1.desc() << std::endl ; 

    //test_get_children(tree0, tree1); 
    test_get_progeny_r(tree0, tree1); 

    return 0 ;  
}
