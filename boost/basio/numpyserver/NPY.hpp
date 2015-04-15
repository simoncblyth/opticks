#pragma once

class G4StepNPY ; 

#include "numpy.hpp"
#include <vector>
#include <string>
#include <sstream>

#include "string.h"
#include "stdlib.h"

class NPY {
   friend class G4StepNPY ; 

   public:
       static std::string path(const char* typ, const char* tag);
       static NPY* load(const char* typ, const char* tag);

       NPY(std::vector<int>& shape, std::vector<float>& data, std::string& metadata) 
         :
         m_shape(shape),
         m_data(data),
         m_metadata(metadata)
      {
         m_len0 = getLength(0);
         m_len1 = getLength(1);
         m_len2 = getLength(2);
      } 

       int getLength(unsigned int n);
       std::string description(const char* msg);

   protected:
       unsigned int       m_len0 ; 
       unsigned int       m_len1 ; 
       unsigned int       m_len2 ; 

   private:
       std::vector<int>   m_shape ; 
       std::vector<float> m_data ; 
       std::string        m_metadata ; 




};


int NPY::getLength(unsigned int n)
{
    return n < m_shape.size() ? m_shape[n] : -1 ;
}


std::string NPY::path(const char* typ, const char* tag)
{
    char* TYP = strdup(typ);
    char* p = TYP ;
    while(*p)
    {
       if( *p >= 'a' && *p <= 'z') *p += 'A' - 'a' ;
       p++ ; 
    } 

    char envvar[64];
    snprintf(envvar, 64, "DAE_%s_PATH_TEMPLATE", TYP ); 
    free(TYP); 

    char* tmpl = getenv(envvar) ;
    if(!tmpl) return "missing-template-envvar" ; 
    
    char path_[256];
    snprintf(path_, 256, tmpl, tag );

    return path_ ;   
}




NPY* NPY::load(const char* typ, const char* tag)
{
    std::string path = NPY::path(typ, tag);

    std::vector<int> shape ;
    std::vector<float> data ;
    std::string metadata = "{}";

    NPY* npy = NULL ;
    try 
    {
        aoba::LoadArrayFromNumpy<float>(path, shape, data );
        npy = new NPY(shape,data,metadata) ;
    } 
    catch(const std::runtime_error& error)
    {
        std::cout << "NPY::load failed " << std::endl ; 
    }


    return npy ;
}


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
    ss << " len0 " << m_len0 ;
    ss << " len1 " << m_len1 ;
    ss << " len2 " << m_len2 ;
    ss << " nfloat " << m_data.size() << " " ;
    ss << m_metadata  ;

    return ss.str();
}
