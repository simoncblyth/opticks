

#include "SArrayDigest.hh"
#include "SDigest.hh"


template <typename T>
std::string SArrayDigest<T>::arraydigest( T* data, unsigned int n )
{
    return SDigest::md5digest( (char*)data, sizeof(T)*n );
}



template class SArrayDigest<float> ;
template class SArrayDigest<int> ;
template class SArrayDigest<unsigned int> ;

 


