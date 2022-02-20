// name=sc4u ; gcc $name.cc -std=c++11 -lstdc++ -I. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name && ipython -i -c "import numpy as np ; a = np.load('/tmp/p.npy') ; print(a.view(np.int8)) " 
#include "sc4u.h"
#include "scuda.h"
#include "squad.h"
#include "NP.hh"


int main(int argc, char** argv)
{
    C4U c4u ; 

    c4u.c4.x = -128 ; 
    c4u.c4.y = 127 ;    // NB int 128  flips to  char -128 
    c4u.c4.z = -128 ; 
    c4u.c4.w = -128 ; 

    unsigned u = c4u.u ;

    std::cout << " c4u   " << C4U_desc(c4u) << std::endl ; 
    std::cout << " c4u.u " << C4U_desc(u)   << std::endl ; 

    std::string s = C4U_name( u, "prefix", '_' ); 

    std::cout << " C4U_name " << s << std::endl ; 



    quad4 p ; 
    p.zero();

    p.q3.u.w = u ; 
   
    NP::Write("/tmp/p.npy", (float*)(&p.q0.f.x), 1, 4, 4 );



    return 0 ; 
}


