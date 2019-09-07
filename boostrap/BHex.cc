/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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
template class BHex<unsigned long long>;




