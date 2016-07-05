#pragma once

#include <map>
#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

template <typename A, typename B>
class BRAP_API  BMap {
   public:
      static void save( std::map<A,B>* , const char* dir, const char* name) ;
      static void save( std::map<A,B>* , const char* path) ;
      static int  load( std::map<A,B>* , const char* dir, const char* name, unsigned int depth=0) ;
      static int  load( std::map<A,B>* , const char* path, unsigned int depth=0 ) ;
      static void dump( std::map<A,B>* , const char* msg="BMap::dump");
   public:
      BMap( std::map<A,B>* mp );
   public:
      void save(const char* dir, const char* name);
      void save(const char* path);
      int load( const char* dir, const char* name, unsigned int depth=0) ;
      int load( const char* path, unsigned int depth=0) ;
      void dump( const char* msg="BMap::dump" ) ;

   private:
      std::map<A,B>* m_map ; 
};

#include "BRAP_TAIL.hh"
 
