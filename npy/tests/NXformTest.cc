#include <cassert>
#include <iostream>

#include "OPTICKS_LOG.hh"

#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"


#include "NXform.hpp"  // header has all the implementation

struct nod 
{
    const nmat4triple* transform ; 
    nod*               parent ; 
};

template struct nxform<nod> ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const nmat4triple* xplus = nmat4triple::make_translate( 100, 0,   0);     
    const nmat4triple* yplus = nmat4triple::make_translate(   0, 100, 0);     
    const nmat4triple* zplus = nmat4triple::make_translate(   0, 0, 100);     
    const nmat4triple* scale = nmat4triple::make_scale(       2, 2, 2);     

    nod a = { scale, NULL } ;
    nod b = { xplus, &a } ;
    nod c = { yplus, &b } ;
    nod d = { zplus, &c } ;

    nxform<nod> xfn(0, false) ; 

    const nmat4triple* global_a = nxform<nod>::make_global_transform(&a) ; 
    const nmat4triple* global_b = nxform<nod>::make_global_transform(&b) ; 
    const nmat4triple* global_c = nxform<nod>::make_global_transform(&c) ; 
    const nmat4triple* global_d = nxform<nod>::make_global_transform(&d) ; 

    std::cout << " global_a " << *global_a << std::endl ; 
    std::cout << " global_b " << *global_b << std::endl ; 
    std::cout << " global_c " << *global_c << std::endl ; 
    std::cout << " global_d " << *global_d << std::endl ; 


    return 0 ; 
}

