#pragma once

#include <vector>
#include <map>
#include "BRAP_API_EXPORT.hh"

template <typename A, typename B>
class BRAP_API BList {
   public:
      BList( std::vector<std::pair<A,B> >* vec );
   public:
      void save(const char* dir, const char* name);
      void save(const char* path);
      void load( const char* dir, const char* name) ;
      void load( const char* path) ;
      void dump( const char* msg="BList::dump" ) ;

   private:
      std::vector<std::pair<A,B> >* m_vec ; 
};







