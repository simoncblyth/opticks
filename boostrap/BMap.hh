#pragma once

#include <map>
#include "BRAP_API_EXPORT.hh"

template <typename A, typename B>
class BRAP_API  BMap {
   public:
      BMap( std::map<A,B>* map );
      void save(const char* dir, const char* name);
     // void load(const char* dir, const char* name);
   private:
      std::map<A,B>* m_map ; 
};

template <typename A, typename B>
inline BMap<A,B>::BMap(std::map<A,B>* map) 
   :
   m_map(map)
{
}  
