// gcc reallocTest.cc -lstdc++ -o /tmp/reallocTest && /tmp/reallocTest 

#include "SProc.hh"

#include <iostream>
#include <iomanip>
#include <string> 
#include <sstream> 
#include <cstring> 
#include <cstdlib> 
#include <cassert> 

template<typename T>
struct A
{
    A(int ni_, int nj_=1, int nk_=1, int nl_=1, int nm_=1) 
        :
        ni(ni_),
        nj(nj_),
        nk(nk_),
        nl(nl_),
        nm(nm_),
        ptr(NULL)
    {
    } 

    unsigned size()     const { return ni*nj*nk*nl*nm ; }
    unsigned itemsize() const { return    nj*nk*nl*nm ; }
 
    void allocate()
    {
        unsigned num_bytes = sizeof(T)*size() ; 
        if(num_bytes > 0)
        {
            ptr = (T*)malloc(num_bytes) ; 
        }
    }
 
    void add(T* src, unsigned num_vals)
    {
        unsigned item = itemsize() ; 
        assert( num_vals % item == 0 ); 
        unsigned num_item = num_vals / item ; 

        unsigned cur_bytes = sizeof(T)*size() ; 
        unsigned add_bytes = sizeof(T)*num_item*item ; 
        unsigned new_bytes = cur_bytes+add_bytes ; 

        if( ptr == NULL )
        {
            ptr = (T*)malloc(add_bytes) ; 
        }
        else
        {
            ptr = (T*)realloc(ptr, new_bytes); 
        }
        memcpy( ptr + ni*item,  src,  add_bytes ); 

        ni += num_item ; 
    }

    void reset()
    {
        free(ptr);
        ptr = NULL ; 
        ni = 0 ; 
    }

    std::string desc() const 
    {
        std::stringstream ss ; 
        ss << " shape (" << ni << "," << nj << "," << nk << "," << nl << "," << nm << ")"  ; 
        std::string s = ss.str(); 
        return s ; 
    }

    int      ni ; 
    int      nj ; 
    int      nk ;
    int      nl ;
    int      nm ;

    T*       ptr ;
};


int main() 
{ 
    unsigned itemsize = 6*4 ; 
    float* f = new float[itemsize]; 
    for(int i=0 ; i < itemsize ; i++) f[i] = float(i); 

    A<float>* a = new A<float>(0, 6, 4 ); 

    float vm0 = SProc::VirtualMemoryUsageMB() ;

    for(int e=0 ; e < 100 ; e++)
    {
        for(int s=0 ; s < 1000 ; s++) a->add(f, itemsize); 
        std::cout << std::setw(4) << e << " : " << a->desc() << std::endl ;  
        a->reset(); 
    }
    float vm1 = SProc::VirtualMemoryUsageMB() ;
    float dv = vm1 - vm0 ; 
    std::cout 
        << " vm0 " << vm0 
        << " vm1 " << vm1 
        << " dv " << dv 
        << std::endl 
        ;

   return 0; 
} 
