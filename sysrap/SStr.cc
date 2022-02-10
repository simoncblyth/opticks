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

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <glm/glm.hpp>


#include "SStr.hh"
#include "SPath.hh"
#include "PLOG.hh"

/**

In [15]: s = "hello"

In [18]: encode_ = lambda s:sum(map(lambda ic:ord(ic[1]) << 8*ic[0], enumerate(s[:8]) ))

In [19]: encode_(s)
Out[19]: 478560413032

In [40]: decode_ = lambda v:"".join(map( lambda c:str(unichr(c)), filter(None,map(lambda i:(v >> i*8) & 0xff, range(8))) ))

In [41]: decode_(478560413032)
Out[41]: 'hello'

Hmm presumably base64 code might do this at a higher level ?

**/




void SStr::Save(const char* path_, const std::vector<std::string>& a, char delim )   // static
{
    int create_dirs = 1 ; // 1:filepath
    const char* path = SPath::Resolve(path_, create_dirs);  // 
    LOG(info) << "SPath::Resolve " << path_ << " to " << path ; 
    std::ofstream fp(path);
    for(std::vector<std::string>::const_iterator i = a.begin(); i != a.end(); ++i) fp << *i << delim ;
}


/**
SStr::Save
-----------

When saving into PWD it is necessary to manually set create_dirs to 0 
otherwise the file is not written but rather an directory is created. TODO: fix this

**/

void SStr::Save(const char* path_, const char* txt, int create_dirs )
{
    const char* path = SPath::Resolve(path_, create_dirs);  // 
    LOG(info) << "SPath::Resolve " << path_ << " to " << path ; 
    std::ofstream fp(path);
    fp << txt ;  
}

const char* SStr::Load(const char* path_ )
{
    int create_dirs = 0 ; // 0:do nothing
    const char* path = SPath::Resolve(path_, create_dirs);  // 
    LOG(info) << "SPath::Resolve " << path_ << " to " << path ; 
    std::ifstream fp(path);

    std::stringstream ss ; 
    ss << fp.rdbuf() ; 
    std::string txt = ss.str(); 
    return strdup(txt.c_str()) ; 
}




void SStr::FillFromULL( char* dest, unsigned long long value, char unprintable)
{
    dest[8] = '\0' ; 
    for( ULL w=0 ; w < 8 ; w++)
    {   
        ULL ullc = (value & (0xffull << w*8)) >> w*8 ;
        char c = static_cast<char>(ullc) ; 
        bool printable = c >= ' ' && c <= '~' ;
        dest[w] = printable ? c : unprintable ; 
    }       
}

const char* SStr::FromULL( unsigned long long value, char unprintable)
{
    assert( sizeof(ULL) == 8 );
    char* s = new char[8+1] ; 
    FillFromULL(s, value, unprintable) ; 
    return s ; 
}


unsigned long long SStr::ToULL( const char* s )
{
    assert( sizeof(ULL) == 8 );

    unsigned len = s ? strlen(s) : 0 ; 
    ULL mxw = len < 8 ? len : 8 ; 

    ULL v = 0ull ;   
    for(ULL w=0 ; w < mxw ; w++)
    {
        ULL c = s[w] ; 
        v |= ( c << 8ull*w ) ; 
    }
    return v ; 
}


template<size_t SIZE>
const char* SStr::Format1( const char* fmt, const char* value )
{
    char buf[SIZE]; 
    size_t cx = snprintf( buf, SIZE, fmt, value );   
    assert( cx < SIZE && "snprintf truncation detected" );  
    return strdup(buf);
}

template<size_t SIZE>
const char* SStr::Format2( const char* fmt, const char* value1, const char* value2 )
{
    char buf[SIZE]; 
    size_t cx = snprintf( buf, SIZE, fmt, value1, value2 );   
    assert( cx < SIZE && "snprintf truncation detected" );  
    return strdup(buf);
}

template<size_t SIZE>
const char* SStr::Format3( const char* fmt, const char* value1, const char* value2, const char* value3 )
{
    char buf[SIZE]; 
    size_t cx = snprintf( buf, SIZE, fmt, value1, value2, value3 );   
    assert( cx < SIZE && "snprintf truncation detected" );  
    return strdup(buf);
}


template const char* SStr::Format1<256>( const char* , const char* );
template const char* SStr::Format2<256>( const char* , const char*, const char* );
template const char* SStr::Format3<256>( const char* , const char*, const char* , const char* );

template const char* SStr::Format1<16>( const char* , const char* );


template<typename T>
const char* SStr::FormatReal(const T value, int w, int p, char fill )
{
    std::stringstream ss ; 
    ss << std::fixed << std::setfill(fill) << std::setw(w) << std::setprecision(p) << value ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()) ; 
} 

template const char* SStr::FormatReal<float>(const float, int, int, char );
template const char* SStr::FormatReal<double>(const double, int, int, char );




bool SStr::Contains( const char* s_ , const char* q_ )
{
    std::string s(s_); 
    std::string q(q_); 
    return s.find(q) != std::string::npos ;
}

bool SStr::EndsWith( const char* s, const char* q)
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ;
}

const char* SStr::StripPrefix_(const char* s, const char* pfx )
{
    const char* ss = pfx && StartsWith(s, pfx ) ? s + strlen(pfx) : s ;
    return strdup(ss); 
}

const char* SStr::StripPrefix(const char* s, const char* pfx0, const char* pfx1, const char* pfx2 )
{
    if(      pfx0 && StartsWith(s,pfx0) )  return StripPrefix_(s, pfx0) ; 
    else if( pfx1 && StartsWith(s,pfx1) )  return StripPrefix_(s, pfx1) ;
    else if( pfx2 && StartsWith(s,pfx2) )  return StripPrefix_(s, pfx2) ;
    return strdup(s); 
}

const char* SStr::MaterialBaseName(const char* s )
{
    return StripPrefix(s, "/dd/Materials/", "_dd_Materials_" );  
}


bool SStr::StartsWith( const char* s, const char* q)
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ;
}

/**
SStr::SimpleMatch
---------------------

Return if the argument string s matches the query string q 
If q ends with '$' require a full match otherwise allow
StartsWith match.

**/
bool SStr::SimpleMatch(const char* s, const char* q )
{
    unsigned ls = strlen(s); 
    unsigned lq = strlen(q); 

    if(ls == 0 ) return false ; 
    if(lq == 0 ) return false ; 

    bool qed = q[lq-1] == '$' || q[lq-1] == '@' ; 
    bool qed_match = 0 == strncmp(s, q, lq - 1) && ls == lq - 1 ;   // exact match up to the dollar 
    return qed ? qed_match : StartsWith(s, q) ;
}



/** 
SStr::Match
-------------

Based on https://www.geeksforgeeks.org/wildcard-character-matching/

See tests/match.cc


The second argument string can contain wildcard tokens:
    
`*` 
     matches with 0 or more of any char (NB '**' not supported)
`?`   
     matches any one character.
`$` or '@'
     when appearing at end of q requires the end of s to match  
   
**/ 
    
bool SStr::Match(const char* s, const char* q) 
{
    if (*q == '\0' && *s == '\0') return true;

    if (*q == '*' && *(q+1) != '\0' && *s == '\0') return false;  // reached end of s but still q chars coming 

    if ( (*q == '$' || *q == '@') && *(q+1) == '\0' && *s == '\0' ) return true ; 
    
    if (*q == '?' || *q == *s) return SStr::Match(s+1, q+1);  // on to next char

    if (*q == '*') return SStr::Match(s, q+1) || SStr::Match(s+1, q);   // '*' can match nothing or anything in s, including literal '*'
          
    return false;
}         

/**

SStr::HasPointerSuffix
-----------------------

Typically see 12 hexdigit pointers, as even though have 64 bits it is normal to only use 48 bits in an address space
    0x7ff46e500520 

But with G4 are seeing only 9 hexdigits ??


**/

bool SStr::HasPointerSuffix( const char* name, unsigned hexdigits )
{
   // eg Det0x110d9a820      why 9 hex digits vs
   //       0x7ff46e500520    
   //
    std::string s(name); 
    unsigned l = s.size() ; 
    if(l < hexdigits+2 ) return false ;
 
    for(unsigned i=0 ; i < hexdigits+2 ; i++)
    {
        char c = s[l-11+i] ; 
        bool ok = false ; 
        switch(i)
        {
            case 0: ok = c == '0' ; break ; 
            case 1: ok = c == 'x' ; break ; 
            default:  ok = ( c >= '0' && c <= '9' ) || ( c >= 'a' && c <= 'f' ) ; break ;   
        }
        if(!ok) return false ; 
    }
    return true  ; 
}


/**
SStr::GetPointerSuffixDigits
------------------------------

Check for hexdigits backwards, until reach first non-hexdigit the 'x'::

   World0x7fc10641cbb0  -> 12 
     Det0x110fa38b0     ->  9
     Hello              -> -1

**/

int SStr::GetPointerSuffixDigits( const char* name )
{
    if( name == NULL ) return -1 ; 
    int l = strlen(name) ; 
    int num = 0 ; 
    for(int i=0 ; i < l ; i++ )  
    {
         char c = *(name + l - 1 - i) ;  // reverse order chars
         bool hexdigit = ( c >= '0' && c <= '9' ) || ( c >= 'a' && c <= 'f' ) ; 
         if(!hexdigit) break ;  
         num += 1 ; 
    } 
    //std::cout << std::endl ; 
    if(l - num - 1 < 0 )  return -1 ; 
    if(l - num - 2 < 0 )  return -1 ; 

    char c1 = *(name + l - num - 1);
    char c2 = *(name + l - num - 2);

    return  c1 == 'x' && c2 == '0'  ?  num : -1 ; 
}

/**
SStr::HasPointerSuffix
------------------------

Returns true when the number of hex digits is within the inclusive min-max range.

**/

bool SStr::HasPointerSuffix( const char* name, unsigned min_hexdigits, unsigned max_hexdigits )
{    
    int num_hexdigits = GetPointerSuffixDigits( name ); 
    return  num_hexdigits > -1 && num_hexdigits >= int(min_hexdigits) && num_hexdigits <= int(max_hexdigits) ; 
}

/**
SStr::TrimPointerSuffix
-------------------------

For an input name such as "SomeName0xdeadbeef"  returns "SomeName"

**/
const char* SStr::TrimPointerSuffix( const char* name )
{
    int num_hexdigits = GetPointerSuffixDigits( name ); // char-by-char look back  
    char* trim = strdup(name); 

    if( num_hexdigits >= 6 && num_hexdigits <= 12 )   // sanity check for the pointer suffix
    {
        int ip = strlen(name) - num_hexdigits - 2 ;  // offset to land on the '0' of "SomeName0xdeadbeef"
        assert( ip >= 0 ); 
        char* p = trim + ip ; 
        assert( *p == '0' ); 
        *p = '\0' ;    // terminate string chopping off the suffix eg "0xdeadbeef"
    }    
    return trim ; 
}





const char* SStr::Concat( const char* a, const char* b, const char* c  )
{
    std::stringstream ss ; 
    if(a) ss << a ; 
    if(b) ss << b ; 
    if(c) ss << c ; 
    std::string s = ss.str();
    return strdup(s.c_str());
}

const char* SStr::Concat( const char* a, unsigned b, const char* c  )
{
    std::stringstream ss ; 
    if(a) ss << a ; 
    ss << b ; 
    if(c) ss << c ; 
    std::string s = ss.str();
    return strdup(s.c_str());
}

const char* SStr::Concat( const char* a, unsigned b, const char* c, unsigned d, const char* e  )
{
    std::stringstream ss ; 

    if(a) ss << a ; 
    ss << b ; 
    if(c) ss << c ; 
    ss << d ; 
    if(e) ss << e ; 

    std::string s = ss.str();
    return strdup(s.c_str());
}


template<typename T>
const char* SStr::Concat_( const char* a, T b, const char* c  )
{
    std::stringstream ss ; 
    if(a) ss << a ; 
    ss << b ; 
    if(c) ss << c ; 
    std::string s = ss.str();
    return strdup(s.c_str());
}



const char* SStr::Replace( const char* s,  char a, char b )
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < strlen(s) ; i++)
    {
        char c = *(s+i) ;   
        ss << ( c == a ? b : c ) ;  
    }
    std::string r = ss.str(); 
    return strdup(r.c_str());
}


/**
SStr::ReplaceEnd
------------------

String s is required to have ending q.
New string n is returned with the ending q replaced with r.

**/

const char* SStr::ReplaceEnd( const char* s, const char* q, const char* r  )
{
    int pos = strlen(s) - strlen(q) ;
    assert( pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 );

    std::stringstream ss ; 
    for(int i=0 ; i < pos ; i++) ss << *(s+i) ;  
    ss << r ; 

    std::string n = ss.str(); 
    return strdup(n.c_str());
}



void SStr::Split( const char* str, char delim,   std::vector<std::string>& elem )
{
    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 
}


int SStr::ISplit( const char* line, std::vector<int>& ivec, char delim )
{
    std::stringstream ss; 
    ss.str(line)  ;

    std::string s;
    while (std::getline(ss, s, delim)) ivec.push_back(std::atoi(s.c_str())) ; 
    
    return ivec.size(); 
}







template const char* SStr::Concat_<unsigned>(           const char* , unsigned           , const char*  );
template const char* SStr::Concat_<unsigned long long>( const char* , unsigned long long , const char*  );
template const char* SStr::Concat_<int>(                const char* , int                , const char*  );
template const char* SStr::Concat_<long>(               const char* , long               , const char*  );




void SStr::ParseGridSpec(  std::array<int,9>& grid, const char* spec)  // static 
{
    int idx = 0 ; 
    std::stringstream ss(spec); 
    std::string s;
    while (std::getline(ss, s, ',')) 
    {   
        std::stringstream tt(s); 
        std::string t;
        while (std::getline(tt, t, ':')) grid[idx++] = std::atoi(t.c_str()) ; 
    }   

    std::stringstream uu ; 
    uu << spec << " : " ;
    for(int i=0 ; i < 9 ; i++) uu << grid[i] << " " ; 
    uu << std::endl ; 

    std::string u = ss.str(); 
    LOG(info) << u ;  
}


void SStr::DumpGrid(const std::array<int,9>& cl)
{   
    int i0 = cl[0] ;
    int i1 = cl[1] ;
    int is = cl[2] ;
    int j0 = cl[3] ;
    int j1 = cl[4] ;
    int js = cl[5] ;
    int k0 = cl[6] ;
    int k1 = cl[7] ;
    int ks = cl[8] ; 

    unsigned num = 0 ; 
    for(int i=i0 ; i < i1 ; i+=is ) 
    for(int j=j0 ; j < j1 ; j+=js ) 
    for(int k=k0 ; k < k1 ; k+=ks ) 
    {
        std::cout << std::setw(2) << num << " (i,j,k) " << "(" << i << "," << j << "," << k << ") " << std::endl ; 
        num += 1 ; 
    }
}






template <typename T>
void SStr::GetEVector(std::vector<T>& vec, const char* key, const char* fallback )
{
    const char* sval = getenv(key); 
    std::stringstream ss(sval ? sval : fallback); 
    std::string s ; 
    while(getline(ss, s, ',')) vec.push_back(ato_<T>(s.c_str()));   
}  

template void  SStr::GetEVector<unsigned>(std::vector<unsigned>& vec, const char* key, const char* fallback  );
template void  SStr::GetEVector<float>(std::vector<float>& vec, const char* key, const char* fallback  );

void SStr::GetEVec(glm::vec3& v, const char* key, const char* fallback )
{   
    std::vector<float> vec ; 
    SStr::GetEVector<float>(vec, key, fallback);
    std::cout << key << SStr::Present(vec) << std::endl ; 
    assert( vec.size() == 3 ); 
    for(int i=0 ; i < 3 ; i++) v[i] = vec[i] ;
}   
    
void SStr::GetEVec(glm::vec4& v, const char* key, const char* fallback )
{   
    std::vector<float> vec ;
    SStr::GetEVector<float>(vec, key, fallback);
    std::cout << key << SStr::Present(vec) << std::endl ;
    assert( vec.size() == 4 );
    for(int i=0 ; i < 4 ; i++) v[i] = vec[i] ; 
}




template <typename T>
std::string SStr::Present(std::vector<T>& vec)
{   
    std::stringstream ss ; 
    for(unsigned i=0 ; i < vec.size() ; i++) ss << vec[i] << " " ;
    return ss.str();
}


template std::string SStr::Present<float>(std::vector<float>& );
template std::string SStr::Present<unsigned>(std::vector<unsigned>& );
template std::string SStr::Present<int>(std::vector<int>& );



template <typename T>
T SStr::GetEValue(const char* key, T fallback) // static 
{   
    const char* sval = getenv(key); 
    T val = sval ? ato_<T>(sval) : fallback ;
    return val ;
}



unsigned SStr::Encode4(const char* s) // static 
{
    unsigned u4 = 0u ; 
    for(unsigned i=0 ; i < std::min(4ul, strlen(s)) ; i++ )
    {
        unsigned u = unsigned(s[i]) ;
        u4 |= ( u << (i*8) ) ;
    }
    return u4 ;
}


template float       SStr::GetEValue<float>(const char* key, float fallback);
template int         SStr::GetEValue<int>(const char* key,   int  fallback);
template unsigned    SStr::GetEValue<unsigned>(const char* key,   unsigned  fallback);
template std::string SStr::GetEValue<std::string>(const char* key,  std::string  fallback);
template bool        SStr::GetEValue<bool>(const char* key,  bool  fallback);


const char* SStr::PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext ) // static
{   
    std::stringstream ss ;
    ss << install_prefix
       << "/ptx/"
       << cmake_target
       << "_generated_"
       << cu_stem
       << cu_ext
       << ".ptx"
       ;
    std::string path = ss.str();
    return strdup(path.c_str());
}


template <typename T>
T SStr::ato_( const char* a )   // static 
{   
    std::string s(a);
    std::istringstream iss(s);
    T v ;  
    iss >> v ;
    return v ;
}


template double   SStr::ato_<double>( const char* ); 
template float    SStr::ato_<float>( const char* ); 
template int      SStr::ato_<int>( const char* ); 
template unsigned SStr::ato_<unsigned>( const char* ); 


void SStr::GridMinMax(const std::array<int,9>& grid, glm::ivec3&mn, glm::ivec3& mx)  // static 
{   
    mn.x = grid[0] ; mx.x = grid[1] ;
    mn.y = grid[3] ; mx.y = grid[4] ;
    mn.z = grid[6] ; mx.z = grid[7] ;
}

void SStr::GridMinMax(const std::array<int,9>& grid, int&mn, int& mx)  // static 
{   
    for(int a=0 ; a < 3 ; a++)
    for(int i=grid[a*3+0] ; i < grid[a*3+1] ; i+=grid[a*3+2] )
    {   
        if( i > mx ) mx = i ;
        if( i < mn ) mn = i ;
    }
    std::cout << "SStr::GridMinMax " << mn << " " << mx << std::endl ;
}



int SStr::AsInt(const char* arg, int fallback )
{
    char* end ;   
    char** endptr = &end ; 
    int base = 10 ;   
    unsigned long ul = strtoul(arg, endptr, base); 
    bool end_points_to_terminator = end == arg + strlen(arg) ;   
    return end_points_to_terminator ? int(ul) : fallback ;  
}


int SStr::ExtractInt(const char* arg, int start, unsigned num, int fallback)
{
    unsigned pos = start < 0 ? strlen(arg) + start : start  ; 
    unsigned len = strlen(arg) ; 
    if(pos > len) return fallback ; 
    if(pos + num > len) return fallback ; 

    std::string s(arg+pos,num) ; 
    return SStr::AsInt(s.c_str(), fallback);
}


/**
SStr::ReplaceChars
--------------------

Duplicate the input string and change all occurences of *repl* chars within the string into *to*

**/

const char* SStr::ReplaceChars(const char* str, const char* repl, char to ) 
{
    char* s = strdup(str); 
    for(unsigned i=0 ; i < strlen(s) ; i++) if(strchr(repl, s[i]) != nullptr) s[i] = to ;  
    return s ; 
}


void SStr::Extract( std::vector<long>& vals, const char* s )
{
    char* s0 = strdup(s); 
    char* p = s0 ; 
    while (*p) 
    {
        if( (*p >= '0' && *p <= '9') || *p == '+' || *p == '-') vals.push_back(strtol(p, &p, 10)) ; 
        else p++ ;
    }
    free(s0); 
}

void SStr::Extract_( std::vector<long>& vals, const char* s )
{
    char* p = const_cast<char*>(s) ; 
    while (*p) 
    {
        if( (*p >= '0' && *p <= '9') || *p == '+' || *p == '-') vals.push_back(strtol(p, &p, 10)) ; 
        else p++ ;
    }
}

void SStr::Extract_( std::vector<float>& vals, const char* s )
{
    char* p = const_cast<char*>(s) ; 
    while (*p) 
    {
        if( (*p >= '0' && *p <= '9') || *p == '+' || *p == '-') vals.push_back(strtof(p, &p)) ; 
        else p++ ;
    }
}





