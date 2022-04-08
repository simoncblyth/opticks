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

int main()
{
    std::vector<quad4> pp(10) ; 

    unsigned hitmask = 0xbeefcafe ; 
    qselector<quad4> hit_selector(hitmask); 

    for(unsigned i=0 ; i < pp.size() ; i++)
    {
        quad4& p = pp[i]; 
        p.zero(); 

        p.q0.f.x = float(i*1000) ; 
        p.q3.u.x = i ; 
        p.q3.u.w = i % 3 == 0 ? hitmask : i  ; 
    }

    unsigned num_p = pp.size(); 
    
    quad4* d_pp ;
    cudaMalloc(&d_pp, num_p*sizeof(quad4));
    cudaMemcpy(d_pp, pp.data(), num_p*sizeof(quad4), cudaMemcpyHostToDevice);

    thrust::device_ptr<quad4> t_pp(d_pp);
    size_t num_hit = thrust::count_if(t_pp, t_pp+num_p , hit_selector );

    std::cout << " num_hit " << num_hit << std::endl ; 

    quad4* d_hit ;   
    cudaMalloc(&d_hit,        num_hit*sizeof(quad4));
    cudaMemset(d_hit, 0,      num_hit*sizeof(quad4));
    thrust::device_ptr<quad4> t_hit(d_hit);  
    thrust::copy_if(t_pp, t_pp+num_p , t_hit, hit_selector );

    std::vector<quad4> hit(num_hit) ; 
    cudaMemcpy(hit.data(), d_hit, num_hit*sizeof(quad4), cudaMemcpyDeviceToHost);

    for(unsigned i=0 ; i < hit.size() ; i++)
    {
        quad4& h = hit[i]; 
        std::cout 
             << " h " 
             << h.q3.u.x << " "  
             << h.q3.u.y << " "  
             << h.q3.u.z << " "  
             << h.q3.u.w << " "  
             << std::endl 
             ; 
    }
    return 0 ; 
}
