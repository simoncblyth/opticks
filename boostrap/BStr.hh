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

#include <vector>
#include <string>
#include <map>

//
// returning std::string is fussy wrt compiler details, making inconvenient 
// ... so prefer to rely on external allocation of output 
// and just fill in the content here, as done in removeField

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BStr {
  public:

     static const char* GetField(const char* name, char delim, int field);
     static unsigned NumField(const char* name, char delim );


     static void rtrim(std::string& s );
     static void replace_all(std::string& s, const std::string& fr, const std::string& to ) ;
     static bool HasChar(const std::string& s, char c);

     static int  atoi( const char* str, int fallback=0 );
     static float atof( const char* str, float fallback=0 );
     static double atod( const char* str, double fallback=0 );
     static const char* utoa( unsigned u, int width=-1, bool zeropad=false );

     static const char* itoa( int i );
     static const char* ctoa( char c );
     static const char* negate(const char* tag);
     static bool listHasKey(const char* dlist, const char* key, const char* delim=",");
     static char* trimPointerSuffixPrefix(const char* origname, const char* prefix);
     static char* DAEIdToG4( const char* daeid, bool trimPtr);

     static bool Contains( const char* s_ , const char* q_, char delim=',' ); 

     static const char* uppercase( const char* str );
     static char* afterLastOrAll(const char* orig, char delim='/');
     static std::string firstChars( const char* orig, unsigned n );

     static std::string patternPickField(std::string str, std::string ptn, int num );
     static std::string ijoin( std::vector<int>& elem, char delim);
     static std::string ujoin( std::vector<unsigned>& elem, char delim);
     static std::string join( const char* a, const char* b, const char* c, const char* d, char delim);

     static int index_first(                                 const std::vector<std::string>& elem, const char* item );
     static int index_all(   std::vector<unsigned>& indices, const std::vector<std::string>& elem, const char* item );


     template<typename T>
     static const char* xtoa( T x );

     template<typename T>
     static T LexicalCast(const char* str, T fallback, bool& badcast ) ;

     template<typename T>
     static T LexicalCast(const char* str) ;


     template<typename T> 
     static unsigned Split(std::vector<T>& elem, const char* line, char delim );

     static void isplit( std::vector<int>& elem, const char* line, char delim );
     static void uslice_append( std::vector<unsigned>& elem, const char* sli, char delim );
     static void usplit( std::vector<unsigned>& elem, const char* line, char delim );
     static void fsplit( std::vector<float>& elem, const char* line, char delim );
     static void fsplitEnv( std::vector<float>& elem, const char* envvar, const char* fallback, char delim=' ' );
     static bool existsEnv( const char* envvar );
     static bool StartsWith( const char* s, const char* q );

     static bool EndsWith( const char* s, const char* q );
     static const char* WithoutEnding(const char* s, const char* q);


     static void split( std::vector<std::string>& elem, const char* line, char delim );
     static std::vector<std::pair<std::string, std::string> > ekv_split( const char* line, char edelim=' ', const char* kvdelim=":" );
     static int ekv_split(std::vector<std::pair<std::string, std::string> >& ekv, const char* line, char edelim=' ', const char* kvdelim=":" );

     static void pair_split(std::vector<std::pair<int, int> >& ekv, const char* line, char edelim=',', const char* pairdelim=":" );  // "1:5,2:10" -> (1,5),(2,10)


     static std::string join(std::vector<std::string>& elem, char delim );
     static void removeField(char* dest, const char* line, char delim, int index );
     static std::string insertField(const char* line, char delim, int index, const char* field);
     static unsigned char* make_uchar4_colors(unsigned int n);

     template <typename T>
     static const char* concat( const char* head, T body, const char* tail );

     static void ReplaceAll(std::string& subject, const char* search, const char* replace) ;

};


#include "BRAP_TAIL.hh"




