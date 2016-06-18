#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

#ifdef _MSC_VER
// 'ViewNPY': object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


int main()
{
    MultiViewNPY mvn("test");

    int N = 10 ;  
    NPY<float>* a = NPY<float>::make(N,1,1,4);
    a->zero();
    for(int i=0 ; i < N ; i++) a->setQuad(i, 0, 0,  1.f,2.f,3.f,4.f ); 
    


    ViewNPY* va = new ViewNPY("va", a,0,0,0);
    va->dump("va"); 
    va->Summary("va");


    ViewNPY* vb = new ViewNPY("vb", a,0,0,0);
    va->dump("vb"); 
    vb->Summary("vb");

 
    mvn.add(va);   
    mvn.add(vb);   

    assert(mvn.getNumVecs() == 2);

    mvn.Summary();


    return 0 ;
}



