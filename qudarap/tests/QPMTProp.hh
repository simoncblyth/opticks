#pragma once

#include "QProp.hh"

template<typename T>
struct QPMTProp
{
    const QProp<T>* rindex ; 
    const NP* thickness ;  

    QPMTProp( const NP* rindex, const NP* thickness );     
    std::string desc() const ; 
};

template<typename T>
inline QPMTProp<T>::QPMTProp( const NP* rindex_ , const NP* thickness_ )
    :
    rindex(QProp<T>::Make3D(rindex_)),
    thickness(thickness_)
{
}

template<typename T>
inline std::string QPMTProp<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "QPMTProp::desc"
       << std::endl
       << "rindex"
       << std::endl
       << rindex->desc()
       << std::endl
       << "thickness"
       << std::endl
       << thickness->desc()
       << std::endl
       ;
    std::string s = ss.str(); 
    return s ;
} 
