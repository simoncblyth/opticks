#include <cmath>
#include "SVec.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    float epsilon = 1e-5 ; 

    std::vector<float> a = {1,2,3,4} ;
    std::vector<float> b = {1.1,2.2,3.3,4.4} ;

    bool dump = true ; 
    float md = SVec<float>::MaxDiff(a, b, dump) ; 
    float md_x = 0.4f ; 
    float md_d = std::fabs(md - md_x) ; 

    assert( md_d < epsilon );


    return 0 ;
}
