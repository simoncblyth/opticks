#include "OPTICKS_LOG.hh"
#include "X4SolidMaker.hh"

void test_CanMake()
{
    bool default_ = X4SolidMaker::CanMake("default") ; 
    assert( default_ == true ); 

    bool uohe = X4SolidMaker::CanMake("UnionOfHemiEllipsoids") ; 
    assert( uohe == true ); 

    bool uohe100 = X4SolidMaker::CanMake("UnionOfHemiEllipsoids100") ; 
    assert( uohe100 == true ); 

    bool other = X4SolidMaker::CanMake("other") ; 
    assert( other == false ); 
}

void test_Make()
{
    const char* qname = "UnionOfHemiEllipsoids-10" ; 
    const G4VSolid* solid = X4SolidMaker::Make( qname ); 
    assert( solid ); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_CanMake();
    test_Make();
  
    return 0 ; 
}
