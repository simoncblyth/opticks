
#include <cuda.h>
#include <cuda_runtime.h>

#include "NP.hh"
#include "SCU_BufferView.h"


template<typename T>
void CollectArr( std::vector<const NP*>& arr )
{
    NP* a = NP::Make<T>(10); 
    T* aa = a->values<T>(); 
    for(int i=0 ; i < a->num_items() ; i++ ) aa[i] = T(i+1) ; 

    NP* b = NP::Make<T>(10); 
    T* bb = b->values<T>(); 
    for(int i=0 ; i < b->num_items() ; i++ ) bb[i] = T((i+1)*10) ; 

    NP* c = NP::Make<T>(10); 
    T* cc = c->values<T>(); 
    for(int i=0 ; i < c->num_items() ; i++ ) cc[i] = T((i+1)*100) ; 

    arr.push_back(a); 
    arr.push_back(b); 
    arr.push_back(c);
} 
 

template <typename T>
void test_SCU_BufferView_host()
{
    std::vector<const NP*> arr ;  
    CollectArr<T>(arr); 

    SCU_BufferView<T> buf = {} ; 
    buf.hostcopy(arr); 

    std::cout << buf.desc() << std::endl ; 
    std::cout << buf.hostdump() << std::endl ; 
}

template <typename T>
void test_SCU_BufferView_dev()
{
    std::vector<const NP*> arr ;  
    CollectArr<T>(arr); 

    SCU_BufferView<T> buf = {} ; 
    buf.upload(arr); 

    std::cout << buf.desc() << std::endl ; 
    std::cout << buf.devdump() << std::endl ; 
}


int main()
{
    test_SCU_BufferView_host<int>() ; 
    test_SCU_BufferView_dev<int>() ; 

    return 0 ; 
}
