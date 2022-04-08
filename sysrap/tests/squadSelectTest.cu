/**
name=squadSelectTest ; nvcc $name.cu -I.. -I/usr/local/cuda -o /tmp/$name && /tmp/$name 

1. setup pp buffer on device
2. thrust::count_if get the number of hits
3. allocate device buffer for hits
4. thrust::copy_if between the pp and hit buffers
5. copy hits down to host   

Q: can thrust::count_if copy from device buffer to host buffer without the intermediate device buffer ?
A: Crovella2016:NO https://stackoverflow.com/questions/36877029/thrust-copy-if-device-to-host

**/


#include <vector>
#include "scuda.h"
#include "squad.h"


#include <thrust/copy.h>
#include <thrust/device_ptr.h>

template<typename T>
unsigned select_count( T* d, unsigned num_d,  qselector<T>& selector )
{
    thrust::device_ptr<T> td(d);
    return thrust::count_if(td, td+num_d , selector );
}

/**
select_copy_device_to_host
----------------------------

This API is awkward because the number selected is not known when making the call.
For example it would be difficult to populate an NP array using this without 
making copies. 

**/

template<typename T>
void select_copy_device_to_host( T** h, unsigned& num_select,  T* d, unsigned num_d, const qselector<T>& selector  )
{
    thrust::device_ptr<T> td(d);
    num_select = thrust::count_if(td, td+num_d , selector );
    std::cout << " num_select " << num_select << std::endl ; 

    T* d_select ;   
    cudaMalloc(&d_select,     num_select*sizeof(T));
    //cudaMemset(d_select, 0,   num_select*sizeof(T));
    thrust::device_ptr<T> td_select(d_select);  

    thrust::copy_if(td, td+num_d , td_select, selector );

    *h = new T[num_select] ; 
    cudaMemcpy(*h, d_select, num_select*sizeof(T), cudaMemcpyDeviceToHost);
}


/**
select_copy_device_to_host_presized
--------------------------------------

The host array must be presized to fit the selection, do so using *select_count* with the same selector. 

**/

template<typename T>
void select_copy_device_to_host_presized( T* h, T* d, unsigned num_d, const qselector<T>& selector, unsigned num_select  )
{
    thrust::device_ptr<T> td(d);

    T* d_select ;   
    cudaMalloc(&d_select,     num_select*sizeof(T));
    //cudaMemset(d_select, 0,   num_select*sizeof(T));
    thrust::device_ptr<T> td_select(d_select);  

    thrust::copy_if(td, td+num_d , td_select, selector );

    cudaMemcpy(h, d_select, num_select*sizeof(T), cudaMemcpyDeviceToHost);
}

void populate( quad4* pp, unsigned num_p, unsigned mask )
{
    for(unsigned i=0 ; i < num_p ; i++)
    {
        quad4& p = pp[i]; 
        p.zero(); 

        p.q0.f.x = float(i*1000) ; 
        p.q3.u.x = i ; 
        p.q3.u.w = i % 3 == 0 ? mask : i  ; 
    }
}

void dump( const quad4* pp, unsigned num_p )
{
    std::cout << " dump num_p:" << num_p << std::endl ; 
    for(unsigned i=0 ; i < num_p ; i++)
    {
        const quad4& h = pp[i]; 
        std::cout 
             << " h " 
             << h.q3.u.x << " "  
             << h.q3.u.y << " "  
             << h.q3.u.z << " "  
             << h.q3.u.w << " "  
             << std::endl 
             ; 
    }
}

template<typename T>
T* upload(const T* h, unsigned num_items )
{
    T* d ;
    cudaMalloc(&d, num_items*sizeof(T));
    cudaMemcpy(d, h, num_items*sizeof(T), cudaMemcpyHostToDevice);
    return d ; 
}

void test_monolithic()
{
    std::vector<quad4> pp(10) ; 
    unsigned mask = 0xbeefcafe ; 
    populate(pp.data(), pp.size(), mask); 

    unsigned num_p = pp.size(); 
    quad4* d_pp = upload(pp.data(), num_p);   

    quad4* hit ; 
    unsigned num_hit ; 
    qselector<quad4> selector(mask); 

    select_copy_device_to_host( &hit, num_hit, d_pp, num_p, selector );  

    dump( hit, num_hit );     
}

void test_presized()
{
    std::vector<quad4> pp(10) ; 
    unsigned mask = 0xbeefcafe ; 
    populate(pp.data(), pp.size(), mask); 

    unsigned num_p = pp.size(); 
    quad4* d_pp = upload(pp.data(), num_p);   

    qselector<quad4> selector(mask); 
    unsigned num_hit = select_count( d_pp, num_p, selector ); 
    std::cout << " num_hit " << num_hit << std::endl ; 

    quad4* hit = new quad4[num_hit] ; 
    select_copy_device_to_host_presized( hit, d_pp, num_p, selector, num_hit ); 

    dump( hit, num_hit );     
}


int main()
{
    //test_monolithic();
    test_presized(); 

    return 0 ; 
}
