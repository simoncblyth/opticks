#pragma once

#include <vector>
#include <string>
#include <sstream>

class NPY {
   public:
       NPY(std::vector<int>& shape, std::vector<float>& data, std::string& metadata) 
         :
         m_shape(shape),
         m_data(data),
         m_metadata(metadata)
      {
      } 

       std::string description(const char* msg);

   private:
       std::vector<int>   m_shape ; 
       std::vector<float> m_data ; 
       std::string        m_metadata ; 

};


std::string NPY::description(const char* msg)
{
    std::stringstream ss ; 

    ss << msg << " (" ;

    for(size_t i=0 ; i < m_shape.size() ; i++)
    {
        ss << m_shape[i]  ;
        if( i < m_shape.size() - 1) ss << "," ;
    }
    ss << ") " ;
    ss << " nfloat " << m_data.size() << " " ;
    ss << m_metadata  ;

    return ss.str();
}
