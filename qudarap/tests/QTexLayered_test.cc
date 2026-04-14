
#include "NP.hh"
#include "QTexLayered.h"


template<typename T>
NP* MakeDemoArray(size_t layers, size_t height, size_t width, size_t payload)
{
    NP* a = NP::Make<T>( layers, height, width, payload );
    T* aa = a->values<T>();

    size_t ni = a->shape[0];
    size_t nj = a->shape[1];
    size_t nk = a->shape[2];

    for(size_t i=0 ; i < ni ; i++)
    {
        T fi = ni == 1 ? 1. : T(i)/T(ni-1) ;
        for(size_t j=0 ; j < nj ; j++)
        {
            T fj = nj == 1 ? 1. : T(j)/T(nj-1) ;
            for(size_t k=0 ; k < nk ; k++)
            {
                T fk = nk == 1 ? 1. : T(k)/T(nk-1) ;
                T value = fi*fj*fk ;
                size_t index = nj*nk*i + nk*j + k ;
                aa[index] = value ;
            }
        }
    }
    return a ;
}


int main()
{
    NP* a = MakeDemoArray<float>(3,1,1024,1);
    QTexLayered<float>* qtl = new QTexLayered<float>(a, 'P');
    std::cout << "qtl.desc " << qtl->desc() << "\n" ;

    return 0 ;
}
