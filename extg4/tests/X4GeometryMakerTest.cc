#include "OPTICKS_LOG.hh"
#include "X4GeometryMaker.hh"

void test_CanMake()
{
    bool default_ = X4GeometryMaker::CanMake("default") ; 
    assert( default_ == true ); 

    bool uohe = X4GeometryMaker::CanMake("UnionOfHemiEllipsoids") ; 
    assert( uohe == true ); 

    bool uohe100 = X4GeometryMaker::CanMake("UnionOfHemiEllipsoids100") ; 
    assert( uohe100 == true ); 

    bool other = X4GeometryMaker::CanMake("other") ; 
    assert( other == false ); 
}

void test_Make()
{
    const char* qname = "UnionOfHemiEllipsoids-10" ; 
    const G4VSolid* solid = X4GeometryMaker::Make( qname ); 
    assert( solid ); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_CanMake();
    test_Make();
  
    return 0 ; 
}
