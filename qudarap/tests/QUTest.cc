#include <cassert>
#include "scuda.h"
#include "QU.hh"


void test_not_set_pointer( int* p )
{
    p = nullptr ; 
}
void test_set_pointer( int** pp )
{
    *pp = nullptr ; 
}
void test_set_pointer()
{
    int i = 101 ; 

    int* p0 = &i ; 
    int* p = &i ; 

    test_not_set_pointer( p ); 
    printf("test_not_set_pointer  p %p \n", p ); 
    assert( p == p0 && "test_not_set_pointer :  expected to NOT set the pointer in the calling scope"); 

    test_set_pointer( &p ); 
    printf("test_set_pointer  p %p \n", p ); 
    assert( p == nullptr && "test_set_pointer : expected to set the pointer in the calling scope, using pointer-to-pointer argument "); 
}


void test_device_free_and_alloc()
{
    std::vector<float> v = {1.f, 2.f, 3.f } ;

    float* d_v = nullptr ; 

    QU::device_free_and_alloc<float>(&d_v, v.size() ); 

    printf(" d_v %p \n", d_v ); 
    assert( d_v ); 

    QU::copy_host_to_device<float>(d_v, v.data(), v.size() );  

    std::vector<float> v2(v.size(), 0.f) ; 
   
    QU::copy_device_to_host<float>(v2.data(), d_v, v2.size() );  

    for(unsigned i=0 ; i < v.size() ; i++ ) assert( v2[i] == v[i] ); 
}


int main(int argc, char** argv)
{
    printf("%s\n",argv[0]); 

    //test_set_pointer(); 
    test_device_free_and_alloc(); 

    return 0 ; 
}
