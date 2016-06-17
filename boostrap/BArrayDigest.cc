

#include "BArrayDigest.hh"
#include "BDigest.hh"


template <typename T>
std::string BArrayDigest<T>::arraydigest( T* data, unsigned int n )
{
    return BDigest::md5digest( (char*)data, sizeof(T)*n );
}



template class BArrayDigest<float> ;
template class BArrayDigest<int> ;
template class BArrayDigest<unsigned int> ;

 


