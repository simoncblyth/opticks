#include <vector>
#include "SStr.hh"
#include "SEnabled.hh"

template<unsigned N>
SEnabled<N>::SEnabled(const char* spec)
    :
    enabled(new std::bitset<N>())
{
    char delim = ',' ; 
    std::vector<int> ivec ; 
    SStr::ISplit( spec, ivec, delim );  

    for(unsigned i=0 ; i < ivec.size() ; i++)
    {
        int idx = ivec[i]; 
        if(idx < 0 ) idx += int(N) ; 
        assert( idx >= 0 && idx < int(N) ); 
        (*enabled)[unsigned(idx)] = true ;  
    }
}



template<unsigned N>
bool SEnabled<N>::isEnabled(unsigned idx) const
{
    return (*enabled)[idx] ; 
}

template struct SEnabled<1>;
template struct SEnabled<2>;
template struct SEnabled<4>;
template struct SEnabled<8>;
template struct SEnabled<16>;
template struct SEnabled<32>;
template struct SEnabled<64>;
template struct SEnabled<128>;
template struct SEnabled<256>;
template struct SEnabled<512>;
template struct SEnabled<1024>;


