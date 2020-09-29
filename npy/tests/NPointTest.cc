// om-;TEST=NPointTest om-t

#include "OPTICKS_LOG.hh"
#include "NPoint.hpp"

void test_ctor()
{
    NPoint* pts = new NPoint(2); 
    pts->set(0, {1.f,0.f,0.f,1.f} ); 
    pts->set(1, {2.f,0.f,0.f,1.f} ); 
    pts->dump("test_ctor"); 

    pts->add({3.f, 0.f, 0.f, 1.f}); 
    pts->dump("test_ctor.after_add"); 

    assert( pts->getNum() == 3 ); 
}

void test_spawnTransformed_identity()
{
     unsigned n = 10 ; 
     NPoint* a = new NPoint(n);
     for(unsigned i=0 ; i < n ; i++) a->set(i, float(i),float(i),float(i),1.f);  
     a->dump("NPointTest.a"); 

     glm::mat4 t0(1.f) ; 
     NPoint* b = a->spawnTransformed(t0); 
     b->dump("NPointTest.b"); 
     assert( NPoint::HasSameDigest(a,b) == true ); 
}

void test_spawnTransformed_translate()
{
     unsigned n = 10 ; 
     NPoint* a = new NPoint(n);
     for(unsigned i=0 ; i < n ; i++) a->set(i, float(i),float(i),float(i),1.f);  
     a->dump("NPointTest.a"); 

     glm::mat4 t1(1.f) ; 
     t1[3] = glm::vec4(100.f, 100.f, 100.f, 1.f ); 

     NPoint* b = a->spawnTransformed(t1); 
     b->dump("NPointTest.b"); 

     assert( NPoint::HasSameDigest(a,b) != true ); 
}





int main(int argc, char** argv)
{
     OPTICKS_LOG(argc, argv); 

     test_ctor();
     //test_spawnTransformed_identity(); 
     //test_spawnTransformed_translate(); 


     return 0 ; 
}
// om-;TEST=NPointTest om-t

