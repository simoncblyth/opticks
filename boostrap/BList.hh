#pragma once

#include <vector>
#include <map>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"


template <typename A, typename B>
class BRAP_API BList {
   public:
      static void save( std::vector<std::pair<A,B> >* vec, const char* dir, const char* name );
      static void save( std::vector<std::pair<A,B> >* vec, const char* path );
      static void load( std::vector<std::pair<A,B> >* vec, const char* dir, const char* name );
      static void load( std::vector<std::pair<A,B> >* vec, const char* path );
      static void dump( std::vector<std::pair<A,B> >* vec, const char* msg="BList::dump");
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


#include "BRAP_TAIL.hh"





