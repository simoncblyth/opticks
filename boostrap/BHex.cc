
#include <sstream>
#include "BHex.hh"

template<typename T>
T BHex<T>::hex_lexical_cast(const char* in) 
{
    T out;
    std::stringstream ss; 
    ss <<  std::hex << in; 
    ss >> out;
    return out;
}

template<typename T>
std::string BHex<T>::as_hex(T in)
{
    BHex<T> val(in);
    return val.as_hex();
}

template<typename T>
std::string BHex<T>::as_dec(T in)
{
    BHex<T> val(in);
    return val.as_dec();
}




template<typename T>
BHex<T>::BHex(T in) : m_in(in) {} 


template<typename T>
std::string BHex<T>::as_hex()
{
    std::stringstream ss; 
    ss <<  std::hex << m_in; 
    return ss.str();
}

template<typename T>
std::string BHex<T>::as_dec() 
{
    std::stringstream ss; 
    ss <<  std::dec << m_in; 
    return ss.str();
}




template class BHex<int>;
template class BHex<unsigned int>;


