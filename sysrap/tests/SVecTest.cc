#include <cmath>
#include "SVec.hh"

#include "PLOG.hh"


void test_MaxDiff()
{
    float epsilon = 1e-5 ; 

    std::vector<float> a = {1,2,3,4} ;
    std::vector<float> b = {1.1,2.2,3.3,4.4} ;

    bool dump = true ; 
    float md = SVec<float>::MaxDiff(a, b, dump) ; 
    float md_x = 0.4f ; 
    float md_d = std::fabs(md - md_x) ; 

    assert( md_d < epsilon );
}

void test_FindIndexOfValue()
{
    std::vector<float> a = {1.1,2.2,3.3,4.4} ;
    int idx ;  

    idx = SVec<float>::FindIndexOfValue( a, 3.3f, 1e-6f );
    assert( idx == 2 );

    idx = SVec<float>::FindIndexOfValue( a, 5.5f, 1e-6f );
    assert( idx == -1 );
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);


    test_MaxDiff();
    test_FindIndexOfValue();



    return 0 ;
}
