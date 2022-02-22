// name=non_recursive_mergesort ; gcc $name.cc -g -std=c++11 -I$OPTICKS_PREFIX/include/SysRap -lstdc++ -o /tmp/$name && /tmp/$name 

#include <iostream>
#include "NP.hh"

/**

https://stackoverflow.com/questions/1557894/non-recursive-merge-sort

**/

template <typename T>
void mergesort(int num, T* a, T* b)
{
    int left, rght, wid, rend;
    int i,j,m,t;

    for (int k=1; k < num; k *= 2 ) 
    {       
        std::cout << " k " << k << std::endl ; 
        for (left=0; left+k < num; left += k*2 ) 
        {
            rght = left + k;        
            rend = rght + k;

            std::cout 
                << " left " << left 
                << " rght " << rght 
                << " rend " << rend 
                << std::endl
                ; 

            if (rend > num) rend = num; 

            m = left; i = left; j = rght; 
            while (i < rght && j < rend) 
            { 
                if (a[i] <= a[j]) 
                {         
                    b[m] = a[i]; i++;
                } 
                else 
                {
                    b[m] = a[j]; j++;
                }
                m++;
            }
            while (i < rght) 
            { 
                b[m]=a[i]; 
                i++; m++;
            }
            while (j < rend) 
            { 
                b[m]=a[j]; 
                j++; m++;
            }
            for (m=left; m < rend; m++) 
            { 
                a[m] = b[m]; 
            }
        }
    }
}


int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp" ; 
    NP* aa = NP::Load(dir, "a.npy") ; 
    if( aa == nullptr ) return 1 ; 
 
    int num = aa->shape[0] ; 

    NP* tt = NP::Make<float>(num) ; 

    std::cout << "aa " << aa->sstr() << std::endl ;
    std::cout << "tt " << tt->sstr() << std::endl ;

    float* a = aa->values<float>(); 
    float* b = tt->values<float>(); 

    std::cout << " a[i] " ; 
    for(int i=0 ; i < 10 ; i++) std::cout << a[i] << " " ; 
    std::cout << std::endl ; 

    mergesort<float>( num, a, b); 

    aa->save(dir, "b.npy");  

    return 0 ; 
}


