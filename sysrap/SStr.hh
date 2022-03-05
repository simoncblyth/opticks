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

#pragma once

/**
SStr
======

Static string utilities.



**/


#include <cstddef>
#include <string>
#include <vector>
#include <array>
#include <glm/fwd.hpp>


#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SStr {

    typedef unsigned long long ULL ;
  public:
      static void Save(const char* path, const std::vector<std::string>& a, char delim='\n' );    
      static void Save(const char* path_, const char* txt, int create_dirs=1 ); // 1:filepath 
      static const char* Load(const char* path_ ); 
  public:
      static void FillFromULL( char* dest, unsigned long long value, char unprintable='.') ; 
      static const char* FromULL(unsigned long long value, char unprintable='.'); 
      static unsigned long long ToULL(const char* s8 ); 


      template <size_t SIZE>
      static const char* Format1( const char* fmt, const char* value );

      template <size_t SIZE>
      static const char* Format2( const char* fmt, const char* value1, const char* value2 );

      template <size_t SIZE>
      static const char* Format3( const char* fmt, const char* value1, const char* value2, const char* value3 );


      template <size_t SIZE>
      static const char* FormatInt( const char* fmt, int value ); 


      template<typename T>
      static const char* FormatReal(const T value, int w, int p, char fill='0' ) ; 


      static bool Contains(const char* s, const char* q ); 
      static bool EndsWith( const char* s, const char* q);
      static bool StartsWith( const char* s, const char* q);
      static bool SimpleMatch(const char* s, const char* q );
      static bool Match(const char* s, const char* q);    // q may contain wildcard chars '?' for 1 character and '*' for several  

      static const char* StripPrefix_(const char* s, const char* pfx0 ); 
      static const char* StripPrefix(const char* s, const char* pfx0, const char* pfx1=nullptr, const char* pfx2=nullptr ); 
      static const char* MaterialBaseName(const char* s ) ; 


      static bool HasPointerSuffix( const char* name, unsigned hexdigits ) ;   // 12 typically, 9 with Geant4 ???
      static bool HasPointerSuffix( const char* name, unsigned min_hexdigits, unsigned max_hexdigits ) ;
      static int  GetPointerSuffixDigits( const char* name );
      static const char* TrimPointerSuffix( const char* name ); 

      static const char* TrimLeading(const char* s); 
      static const char* TrimTrailing(const char* s); 
      static const char* Trim(const char* s); 

      static const char* HeadFirst(const char* s, char c); 
      static const char* HeadLast( const char* s, char c); 

      static const char* Concat( const char* a, const char* b, const char* c=NULL  );
      static const char* Concat( const char* a, unsigned b   , const char* c=NULL  );
      static const char* Concat( const char* a, unsigned b, const char* c, unsigned d, const char* e  ) ; 

      template <typename T>
      static const char* Concat_( const char* a, T b   , const char* c  );


    

      static const char* Replace( const char* s,  char a, char b ); 
      static const char* ReplaceEnd( const char* s, const char* q, const char* r  ); 

      static void Split( const char* str, char delim,   std::vector<std::string>& elem ) ;
      static int  ISplit( const char* line, std::vector<int>& ivec, char delim ) ;

      static void ParseGridSpec(  std::array<int,9>& grid, const char* spec); 
      static void DumpGrid(      const std::array<int,9>& grid ) ;


      static void GetEVec(glm::vec3& v, const char* key, const char* fallback );
      static void GetEVec(glm::vec4& v, const char* key, const char* fallback );
    
      template <typename T>
      static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );

      template <typename T>
      static std::string Present(std::vector<T>& vec);


      static const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" );

      static void GridMinMax( const std::array<int,9>& grid, int& mn, int& mx ) ;
      static void GridMinMax( const std::array<int,9>& grid, glm::ivec3& mn, glm::ivec3& mx) ;

      static unsigned Encode4(const char* s); 

      template <typename T>
      static T ato_( const char* a );
 
      template <typename T>
      static T GetEValue(const char* key, T fallback);  


      static int AsInt(const char* arg, int fallback=-1 ) ; 
      static int ExtractInt(const char* arg, int start, unsigned num, int fallback=-1) ;


      static const char* ReplaceChars(const char* str, const char* repl="(),[]", char to=' ') ; 

      static long ExtractLong( const char* s, long fallback ); 
      static void Extract( std::vector<long>& vals, const char* s ); 
      static void Extract_( std::vector<long>& vals, const char* s ); 
      static void Extract_( std::vector<float>& vals, const char* s ); 


};



