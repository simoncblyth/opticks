// ./qgs.sh 

#include <iostream>
#include "scuda.h"
#include "qgs.h"

void test_qg_union()
{
    quad6 q ; 
    q.q0.f = make_float4( 0.1f, 0.2f, 0.3f, 0.4f ); 
    q.q1.f = make_float4( 1.1f, 1.2f, 1.3f, 1.4f ); 
    q.q2.f = make_float4( 2.1f, 2.2f, 2.3f, 2.4f ); 
    q.q3.f = make_float4( 3.1f, 3.2f, 3.3f, 3.4f ); 
    q.q4.f = make_float4( 4.1f, 4.2f, 4.3f, 4.4f ); 
    q.q5.f = make_float4( 5.1f, 5.2f, 5.3f, 5.4f ); 
    const quad6* src = &q ; 

    QG qg ; 
    qg.load(src, 0); 

    GS& g = qg.g ;  

    std::cout << "g.sc0.ScintillationTime [" << g.sc0.ScintillationTime << "]" << std::endl ; 
    std::cout << "g.sc1.ScintillationTime [" << g.sc1.ScintillationTime << "]" << std::endl ; 
    std::cout << "g.ck.maxSin2 [" << g.ck.maxSin2 << "]" << std::endl ; 
}




int main(int argc, char** argv)
{
    test_qg_union(); 

    return 0 ; 
}


