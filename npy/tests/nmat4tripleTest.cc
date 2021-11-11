#include "OPTICKS_LOG.hh"
#include "GLMFormat.hpp"
#include "nmat4triple.hpp"

void dump(const char* msg, const nmat4triple* tr )
{
    std::cout 
        << msg 
        << std::endl 
        << gpresent("t", tr->t )
        << std::endl 
        << gpresent("v", tr->v )
        << std::endl 
        << gpresent("q", tr->q )
        << std::endl 
        ;
}



/**
Scaling and Translation transform order DOES MATTER
-------------------------------------------------------

Because the translation is applied to the scaled coordinates 
when scaling gets done first (as is standard). 

https://stackoverflow.com/questions/53176759/why-does-order-of-scaling-and-translation-matter-for-the-model-matrix

Translation first:

   | Sx 0  0  0 |   | 1 0 0 Tx |     | Sx 0  0  Sx * Tx |
   | 0  Sy 0  0 | * | 0 1 0 Ty |  =  | 0  Sy 0  Sy * Ty |
   | 0  0  Sz 0 |   | 0 0 1 Tz |     | 0  0  Sz Sz * Tz |
   | 0  0  0  1 |   | 0 0 0 1  |     | 0  0  0     1    |

Scaling first:

   | 1 0 0 Tx |   | Sx 0  0  0 |     | Sx 0  0  Tx |
   | 0 1 0 Ty | * | 0  Sy 0  0 |  =  | 0  Sy 0  Ty |
   | 0 0 1 Tz |   | 0  0  Sz 0 |     | 0  0  Sz Tz |
   | 0 0 0 1  |   | 0  0  0  1 |     | 0  0  0  1  |

**/
void test_scale_translate(const  nmat4triple* s, const nmat4triple* t)
{
    dump("s",s); 
    dump("t",t); 

    const nmat4triple* st_false = nmat4triple::product( s, t, false ); 
    dump("nmat4triple::product(s,t,false)\n",st_false); 

    const nmat4triple* st_true = nmat4triple::product( s, t, true ); 
    dump("nmat4triple::product(s,t,true)\n",st_true); 
}

void test_scale_rotate_translate(const  nmat4triple* s, const nmat4triple* r, const nmat4triple* t)
{
    dump("s",s); 
    dump("r",r); 
    dump("t",t); 

    const nmat4triple* srt_false = nmat4triple::product( s, r, t, false ); 
    dump("nmat4triple::product(s,r,t,false)\n",srt_false); 

    const nmat4triple* srt_true = nmat4triple::product( s, r, t, true ); 
    dump("nmat4triple::product(s,r,t,true)\n",srt_true); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const nmat4triple* s = nmat4triple::make_scale(    1.5f, 1.5f,  0.5f ); 
    const nmat4triple* r = nmat4triple::make_rotate(   0.f, 0.f , 1.0,  45.f ); 
    const nmat4triple* t = nmat4triple::make_translate( 0.f, 0.f , -50.f ); 
    assert( s && t && r); 

    test_scale_translate(s, t); 
    //test_scale_rotate_translate(s, r, t); 

    return 0 ; 
}


