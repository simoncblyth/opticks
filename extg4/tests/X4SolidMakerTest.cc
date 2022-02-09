#include "OPTICKS_LOG.hh"
#include "X4SolidMaker.hh"

void test_CanMake()
{
    bool orb_ = X4SolidMaker::CanMake("Orb") ; 
    assert( orb_ == true ); 

    bool uohe = X4SolidMaker::CanMake("UnionOfHemiEllipsoids") ; 
    assert( uohe == true ); 

    bool uohe100 = X4SolidMaker::CanMake("UnionOfHemiEllipsoids100") ; 
    assert( uohe100 == true ); 

    bool other = X4SolidMaker::CanMake("other") ; 
    assert( other == false ); 
}

void test_Make(const char* qname_)
{
    const char* qname = qname_ ? qname_ : "UnionOfHemiEllipsoids-10" ;
    std::string meta ; 
    const G4VSolid* solid = X4SolidMaker::Make( qname, meta  ); 
    LOG(info) << " qname " << qname << " solid " << solid ;  
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_CanMake();
    test_Make(nullptr);

    for(int i=1 ; i < argc ; i++) test_Make( argv[i] ); 
  
    return 0 ; 
}
