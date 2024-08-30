#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>

struct sstr
{
    enum { MATCH_ALL, MATCH_START, MATCH_END } ; 

    static void Write(const char* path, const char* txt ); 

    static bool Match( const char* s, const char* q, bool starting=true ); 
    static bool Match_(     const char* s, const char* q, int mode); 
    static bool MatchAll(   const char* s, const char* q); 
    static bool MatchStart( const char* s, const char* q); 
    static bool StartsWith( const char* s, const char* q); 
    static bool MatchEnd(   const char* s, const char* q); 
    static bool EndsWith(   const char* s, const char* q); 

    static constexpr const char* AZaz = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" ; 
    static bool StartsWithLetterAZaz(const char* q ); 

    static bool Contains(   const char* s_ , const char* q_); 

    static const char* TrimLeading(const char* s);
    static const char* TrimTrailing(const char* s);
    static const char* Trim(const char* s); // both leading and trailing 

    static std::string TrimString(const std::string& str, const std::string& whitespace=" \t" );

    static bool        HasTail(const std::vector<std::string>& names, const char* end="0x");   
    static size_t      CountTail(const std::vector<std::string>& names, const char* end="0x");
    static bool        HasTail(const std::string& name, const char* end="0x");

    static std::string StripTail(const std::string& name, const char* end="0x"); 
    static std::string StripComment(const std::string& name); 

    static std::string StripTail(const char* name, const char* end="0x"); 
    static void        StripTail(       std::vector<std::string>& dst, const std::vector<std::string>& src, const char* end="0x");  
    static void        StripTail_Unique(std::vector<std::string>& dst, const std::vector<std::string>& src, const char* end="0x");  
    static std::string DescKeySrc(const std::vector<std::string>& key, const std::vector<std::string>& src ); 

    static std::string RemoveSpaces(const char* s);  
    static std::string Replace(const char* s, char q, char r); 



    static const char* ReplaceChars(const char* str, const char* repl, char to ); 
    static std::string ReplaceEnd_( const char* s, const char* q, const char* r  ); 
    static const char* ReplaceEnd(  const char* s, const char* q, const char* r  ); 

    static void PrefixSuffixParse(std::vector<std::string>& elem, const char* prefix, const char* suffix, const char* lines); 
    static void Split(     const char* str, char delim,   std::vector<std::string>& elem ); 
    static void SplitTrim( const char* str, char delim,   std::vector<std::string>& elem ); 
    static void SplitTrimSuppress( const char* str, char delim,   std::vector<std::string>& elem ); 

    static void Chop( std::pair<std::string, std::string>& head__tail, const char* delim, const char* str ); 
    static void chop( char** head, char** tail, const char* delim, const char* str ); 

    template<typename T>
    static void split(std::vector<T>& elem, const char* str, char delim  ); 

    template<typename ... Args>
    static std::string Format_( const char* fmt, Args ... args ); 

    static std::string FormatIndexDefault_( int idx, const char* hdr=nullptr  );  // "A000" "A001" "A002" ... 
    static std::string FormatIndex_( int idx, char prefix, int wid, const char* hdr ); 
    static const char* FormatIndex(  int idx, char prefix, int wid, const char* hdr ); 




    template<typename ... Args>
    static std::string Join( const char* delim, Args ... args ); 

    template<typename ... Args>
    static std::string Concat_( Args ... args ); 

    template<typename ... Args>
    static const char* Concat( Args ... args ); 



    static bool Blank(const char* s ); 
    static bool All(const char* s, char q ); 
    static unsigned Count(const char* s, char q ); 

    template<typename T>
    static T To(const char* arg) ; 

    static int AsInt(const char* arg, int fallback=-1 ) ; 
    static const char* ParseStringIntInt( const char* triplet, int& y, int& z, char delim=':' ); 

    template<typename T>
    static void ParsePair( const char* txt, T& x, T& y, char delim=':' ); 

    static bool IsInteger(const char* str); 
    static bool IsWhitespace(const std::string& s ); 

    static bool isdigit_(char c );
    static bool isalnum_(char c );
    static bool isupper_(char c );
    static bool islower_(char c );


    template<typename T>
    static T ParseIntSpec( const char* spec, T& scale ); 

    template<typename T>
    static T ParseInt( const char* spec ); 

    template<typename T>
    static void ParseIntSpecList( std::vector<T>& ii, const char* spec, char delim=',' ); 

    template<typename T>
    static void ParseScale( const char* spec, T& scale ); 

    template<typename T>
    static std::vector<T>* ParseIntSpecList( const char* spec, char delim=',' ) ; 


    static void truncated_copy( char* dst, const char* src, int dst_size ); 
    static void Extract( std::vector<long>& vals, const char* s ); 
    static long ExtractLong( const char* s, long fallback ); 

};

inline void sstr::Write(const char* path, const char* txt )
{
    std::ofstream fp(path);
    fp << txt ;  
}

inline bool sstr::Match( const char* s, const char* q, bool starting )
{
    return starting ? MatchStart(s, q) : MatchAll(s, q) ;
}

inline bool sstr::Match_( const char* s, const char* q, int mode )
{
    bool ret = false ; 
    switch(mode)
    {
        case MATCH_ALL:    ret = MatchAll(  s, q) ; break ; 
        case MATCH_START:  ret = MatchStart(s, q) ; break ; 
        case MATCH_END:    ret = MatchEnd(  s, q) ; break ; 
    }
    return ret ;
}

inline bool sstr::MatchAll( const char* s, const char* q)
{
    return s && q && strcmp(s, q) == 0 ; 
}

/**
sstr::MatchStart (NB this can replace SStr::StartsWith with same args)
-------------------------------------------------------------------------

The 2nd query string must be less than or equal to the length of the first string and 
all the characters of the query string must match with the first string in order 
to return true.::

                    s              q  
   sstr::MatchStart("hello/world", "hello") == true 

**/
inline bool sstr::MatchStart( const char* s, const char* q)
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ;
}
inline bool sstr::StartsWith( const char* s, const char* q)  // synonym for sstr::MatchStart
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ;
}


inline bool sstr::MatchEnd( const char* s, const char* q)
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ;
}
inline bool sstr::EndsWith( const char* s, const char* q)
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ;
}
inline bool sstr::StartsWithLetterAZaz(const char* q )
{
    const char* p = q != nullptr && strlen(q) > 0 ? strchr(AZaz, q[0]) : nullptr ; 
    return p != nullptr ;  
}




inline bool sstr::Contains( const char* s_ , const char* q_ )
{       
    std::string s(s_); 
    std::string q(q_);  
    return s.find(q) != std::string::npos ;
}


inline const char* sstr::TrimLeading(const char* s)
{
    char* p = strdup(s); 
    while( *p && ( *p == ' ' || *p == '\n' )) p++ ; 
    return p ; 
}
inline const char* sstr::TrimTrailing(const char* s) // reposition null terminator to skip trailing whitespace 
{
    char* p = strdup(s); 
    char* e = p + strlen(p) - 1 ;  
    while(e > p && ( *e == ' ' || *e == '\n' )) e-- ;
    e[1] = '\0' ;
    return p ;  
}
inline const char* sstr::Trim(const char* s)  // trim leading and trailing whitespace 
{
    char* p = strdup(s); 
    char* e = p + strlen(p) - 1 ; 
    while(e > p && ( *e == ' ' || *e == '\n' )) e-- ;
    *(e+1) = '\0' ;
    while( *p && ( *p == ' ' || *p == '\n')) p++ ; 
    return p ; 
}

inline std::string sstr::TrimString(const std::string& str, const std::string& whitespace )
{
    const auto beg = str.find_first_not_of(whitespace); 
    if(beg == std::string::npos) return "" ; 
    const auto end = str.find_last_not_of(whitespace); 
    const auto rng = end - beg + 1 ; 
    return str.substr(beg, rng);     
}


   
inline bool sstr::HasTail(const std::vector<std::string>& names, const char* end)
{
    size_t num_names = names.size() ; 
    size_t num_tail = CountTail(names, end ); 
    return num_names == num_tail ;    
}

inline size_t sstr::CountTail(const std::vector<std::string>& names, const char* end)
{
    size_t num_names = names.size() ; 
    size_t count = 0 ; 
    for(unsigned i=0 ; i < num_names ; i++) if(HasTail(names[i], end)) count += 1 ; 
    return count ; 
}

inline bool sstr::HasTail(const std::string& name, const char* end)
{
    bool found = name.find(end) != std::string::npos ; 
    return found ;  
}

inline std::string sstr::StripTail(const std::string& name, const char* end)  // static 
{
    std::string sname = name.substr(0, name.find(end)) ;
    return sname ;
}

inline std::string sstr::StripComment(const std::string& name)  // static 
{
    const char* end = "#" ; // 1st  
    std::string sname = name.substr(0, name.find(end)) ;
    return TrimString(sname) ; 
}



inline std::string sstr::StripTail(const char* name_, const char* end)  // static 
{
    std::string name(name_); 
    return StripTail(name, end) ; 
}

inline void sstr::StripTail(std::vector<std::string>& dst, const std::vector<std::string>& src, const char* end)  // static 
{
    int num_src = src.size(); 
    for(int i=0 ; i < num_src ; i++ )
    {
        const std::string&  _src = src[i] ;
        std::string _dst = StripTail(_src, end); 
        dst.push_back(_dst) ;   
    }
}

/**
sstr::StripTail_Unique
-----------------------

When the stripped name is unique amoungst all 
the stripped names use it as the key otherwise 
try different numbered suffix _0 _1 _2 _3 until 
a unique key amoungst the keys is found.

**/

inline void sstr::StripTail_Unique( std::vector<std::string>& keys, const std::vector<std::string>& src, const char* end )
{
    std::vector<std::string> stripped ; 
    StripTail( stripped, src, end ); 

    int num_src = src.size(); 
    int num_stripped = stripped.size(); 
    assert( num_src == num_stripped ); 

    for(int i=0 ; i < num_src ; i++)
    {
        const char* cand0 = stripped[i].c_str(); 
        int count0 = std::count( stripped.begin(), stripped.end(), cand0 ); 
        assert( count0 >= 1 ); 
        if( count0 == 1 )
        {
            keys.push_back(cand0);  
        }
        else
        {
            for(int j=0 ; j < 1000000 ; j++ )
            {
                std::string cand = Format_("%s_%d", cand0, j ); 
                int count_key = std::count( keys.begin(), keys.end(), cand.c_str() ); 
                assert( count_key == 0 || count_key == 1 ); 
                if( count_key == 0 ) 
                {
                    keys.push_back(cand); 
                    break ; 
                }
            }
        }
    }
}

inline std::string sstr::DescKeySrc(const std::vector<std::string>& key, const std::vector<std::string>& src )
{
    std::stringstream ss ;  
    ss << "sstr::DescKeySrc" << std::endl ; 
    int num_src = src.size(); 
    int num_key = key.size(); 
    assert( num_src == num_key );  

    for(int i=0 ; i < num_src ; i++)
    {
        ss << std::setw(4) << i 
           << " : "
           << std::setw(50) << src[i]
           << " : "
           << std::setw(50) << key[i]
           << std::endl 
           ;
    }
    std::string str = ss.str(); 
    return str ;
}


inline std::string sstr::RemoveSpaces(const char* s) // static
{
    std::stringstream ss ;  
    for(int i=0 ; i < int(strlen(s)) ; i++) if(s[i] != ' ') ss << s[i] ;   
    std::string str = ss.str(); 
    return str ; 
}
inline std::string sstr::Replace(const char* s, char q, char r) // static
{
    std::stringstream ss ;  
    for(int i=0 ; i < int(strlen(s)) ; i++) ss << ( s[i] == q ? r : s[i] ) ;   
    std::string str = ss.str(); 
    return str ; 
}



inline const char* sstr::ReplaceChars(const char* str, const char* repl, char to )
{
    char* s = strdup(str);  
    for(unsigned i=0 ; i < strlen(s) ; i++) if(strchr(repl, s[i]) != nullptr) s[i] = to ;
    return s ; 
}   




/**
sstr::ReplaceEnd_
------------------

String s is required to have ending q.
New string n is returned with the ending q replaced with r.

**/

inline std::string sstr::ReplaceEnd_( const char* s, const char* q, const char* r  )
{
    int pos = strlen(s) - strlen(q) ;
    assert( pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ); // check q is at end of s 

    std::stringstream ss ; 
    for(int i=0 ; i < pos ; i++) ss << *(s+i) ;  // copy front of s 
    ss << r ;    // replace the end 

    std::string str = ss.str(); 
    return str ; 
}

inline const char* sstr::ReplaceEnd( const char* s, const char* q, const char* r  )
{
    std::string str = ReplaceEnd_(s, q, r); 
    return strdup(str.c_str());
}






inline void sstr::PrefixSuffixParse(std::vector<std::string>& elem, const char* prefix, const char* suffix, const char* lines)
{
    std::stringstream ss;  
    ss.str(lines)  ;
    std::string s;
    while (std::getline(ss, s, '\n')) 
    {
        if(s.empty()) continue ;  
        const char* l = s.c_str(); 
        bool has_prefix = strlen(l) > strlen(prefix) && strncmp(l, prefix, strlen(prefix)) == 0 ; 
        bool has_suffix = strlen(l) > strlen(suffix) && strncmp(l+strlen(l)-strlen(suffix), suffix, strlen(suffix)) == 0 ; 
        //std::cout << "[" << l << "]"<< " has_prefix " << has_prefix << " has_suffix " << has_suffix << std::endl ; 
        if(has_prefix && has_suffix)
        {
              int count = strlen(l) - strlen(prefix) - strlen(suffix) ; 
              std::string sub = s.substr(strlen(prefix), count ); 
              //std::cout << " count " << count << " sub [" << sub << "]" << std::endl ; 
              elem.push_back(sub); 
        }
    }
}


inline void sstr::Split( const char* str, char delim,   std::vector<std::string>& elem )
{
    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 
}

inline void sstr::SplitTrim( const char* str, char delim,   std::vector<std::string>& elem  )
{
    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(Trim(s.c_str())) ; 
}

/**
sstr::SplitTrimSuppress
------------------------

* when str contains '\n' the argument delim is overridden to become '\n'
* elem that start with # are skipped 

**/

inline void sstr::SplitTrimSuppress( const char* str, char delim,   std::vector<std::string>& elem  )
{
    bool is_multiline = Contains(str,"\n");
    char udelim = is_multiline ? '\n' : delim ; 

    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, udelim)) 
    {
        const char* t = Trim(s.c_str());
        if(t && strlen(t) > 0) 
        {
            if(t[0] == '#') continue ;  
            elem.push_back(t) ;
        } 
    }
}




template<typename T>
inline void sstr::split( std::vector<T>& elem, const char* str, char delim )
{
    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, delim)) 
    {
        std::istringstream iss(s);
        T v ;  
        iss >> v ;
        elem.push_back(v) ; 
    }
}





inline void sstr::Chop( std::pair<std::string, std::string>& head__tail, const char* delim, const char* str )
{
    char* head = strdup(str); 
    char* p = strstr(head, delim);  // pointer to first occurence of delim in str or null if not found
    if(p) p[0] = '\0' ; 
    const char* tail = p ? p + strlen(delim)  : nullptr ; 
    head__tail.first = head ; 
    head__tail.second = tail ? tail : ""  ; 
}  

inline void sstr::chop( char** head, char** tail, const char* delim, const char* str )
{
    *head = strdup(str); 
    char* p = strstr(*head, delim);  // pointer to first occurence of delim in str or null if not found
    if(p) p[0] = '\0' ; 
    *tail = p ? p + strlen(delim) : nullptr ; 
}

/**
sstr::Format_
---------------

See sysrap/tests/StringFormatTest.cc


**/

template<typename ... Args>
inline std::string sstr::Format_( const char* fmt, Args ... args )
{
    int sz = std::snprintf( nullptr, 0, fmt, args ... ) + 1 ; // +1 for null termination
    assert( sz > 0 );   
    std::vector<char> buf(sz) ;    
    std::snprintf( buf.data(), sz, fmt, args ... );
    return std::string( buf.begin(), buf.begin() + sz - 1 );  // exclude null termination 
}

template std::string sstr::Format_( const char*, const char*, int, int ); 


inline std::string sstr::FormatIndexDefault_( int idx, const char* hdr )
{
    char prefix = 'A' ; 
    int wid = 3 ; 
    return FormatIndex_(idx,  prefix, wid, hdr ); 
}

inline std::string sstr::FormatIndex_( int idx, char prefix, int wid, const char* hdr )
{
    assert( prefix == '\0' || prefix == 'A' || prefix == 'B' ); 
    assert( idx >= 0 ); 

    std::stringstream ss ;  
    if(hdr) ss << hdr ; 

    //if(prefix) ss << ( idx == 0 ? "z" : ( idx < 0 ? "n" : "p" ) ) ; 
    if(prefix != '\0') ss << prefix  ; 

    ss << std::setfill('0') << std::setw(wid) << std::abs(idx) ; 
    std::string str = ss.str(); 
    return str ; 
}

/**
sstr::FormatIndex
-------------------

prefix:A wid:3

+-----+--------+
| idx |  val   |
+=====+========+
|  1  | A001   |
+-----+--------+
|  -1 | *FAIL* |
+-----+--------+
|  0  | A000   |
+-----+--------+

**/

inline const char* sstr::FormatIndex( int idx, char prefix, int wid, const char* hdr )
{
    std::string str = FormatIndex_(idx, prefix, wid, hdr ); 
    return strdup(str.c_str()); 
}




template<typename ... Args>
inline std::string sstr::Join( const char* delim, Args ... args_ )
{
    std::vector<const char*> args = {args_ ...};
    int num_args = args.size() ;  
    std::stringstream ss ; 
    for(int i=0 ; i < num_args ; i++) ss << ( args[i] ? args[i] : "" ) << ( i < num_args - 1 ? delim : "" ) ; 
    std::string str = ss.str(); 
    return str ; 
}
template std::string sstr::Join( const char*, const char*, const char* ); 
template std::string sstr::Join( const char*, const char*, const char*, const char*  ); 
template std::string sstr::Join( const char*, const char*, const char*, const char*, const char* ); 





template<typename ... Args>
inline std::string sstr::Concat_( Args ... args_ )
{
    std::vector<const char*> args = {args_ ...};
    int num_args = args.size() ;  
    std::stringstream ss ; 
    for(int i=0 ; i < num_args ; i++) ss << ( args[i] ? args[i] : "" ) ; 
    std::string str = ss.str(); 
    return str ;
}

template std::string sstr::Concat_( const char*, const char* ); 
template std::string sstr::Concat_( const char*, const char*, const char* ); 
template std::string sstr::Concat_( const char*, const char*, const char*, const char*  ); 
template std::string sstr::Concat_( const char*, const char*, const char*, const char*, const char* ); 





template<typename ... Args>
inline const char* sstr::Concat( Args ... args )
{
    std::string str = Concat_(std::forward<Args>(args)... ); 
    return strdup(str.c_str()); 
}

template const char* sstr::Concat( const char*, const char* ); 
template const char* sstr::Concat( const char*, const char*, const char* ); 
template const char* sstr::Concat( const char*, const char*, const char*, const char*  ); 
template const char* sstr::Concat( const char*, const char*, const char*, const char*, const char* ); 








inline bool sstr::Blank( const char* s )
{
   unsigned n = strlen(s) ; 
   return n == 0 || All(s, ' ') ; 
}

inline bool sstr::All( const char* s , char q )
{
   unsigned n = strlen(s) ; 
   return n > 0 && Count(s, q) == n ; 

}
inline unsigned sstr::Count( const char* s , char q )
{
   unsigned n = strlen(s) ; 
   unsigned count = 0 ; 
   for(unsigned i=0 ; i < n ; i++) if( s[i] == q ) count += 1 ; 
   return count ;  
}

template<typename T>
inline T sstr::To(const char* arg )
{
    std::string str(arg);
    std::istringstream iss(str);
    T v ;    
    iss >> v ;  
    return v ;  
}

// specialization for std::string as the above truncates at the first blank in the string, see tests/NP_set_meta_get_meta_test.cc  
template<> inline std::string sstr::To(const char* a )  
{
    std::string s(a); 
    return s ;  
}




inline int sstr::AsInt(const char* arg, int fallback )
{
    char* end ;   
    char** endptr = &end ; 
    int base = 10 ;   
    unsigned long ul = strtoul(arg, endptr, base); 
    bool end_points_to_terminator = end == arg + strlen(arg) ;   
    return end_points_to_terminator ? int(ul) : fallback ;  
}


inline const char* sstr::ParseStringIntInt( const char* triplet, int& y, int& z, char delim )
{
    std::stringstream ss; 
    ss.str(triplet)  ;
    std::string s;
    std::vector<std::string> elem ; 
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 
    assert(elem.size() == 3 ); 
    y = AsInt( elem[1].c_str() ); 
    z = AsInt( elem[2].c_str() ); 
    return strdup(elem[0].c_str()); 
}


template<typename T>
inline void sstr::ParsePair( const char* txt, T& x, T& y, char delim )
{
    std::stringstream ss; 
    ss.str(txt)  ;
    std::string s;
    std::vector<std::string> elem ; 
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 
    int num_elem = elem.size();  
    bool expect = num_elem == 2 ;
    if(!expect) std::cerr 
        << "sstr::ParsePair"
        << " txt [" << ( txt ? txt : "-" ) << "]"
        << " delim " << delim 
        << " num_elem " << num_elem 
        << std::endl
        ;

    assert(expect); 
    x = To<T>( elem[0].c_str() ); 
    y = To<T>( elem[1].c_str() ); 
}

inline bool sstr::IsInteger(const char* str)
{
    if(!str) return false ; 
    if(strlen(str)==0) return false ; 

    std::string s(str);
    return s.find_first_not_of("0123456789") == std::string::npos ; 
}

inline bool sstr::IsWhitespace(const std::string& str )
{
    return str.find_first_not_of(" \t\n\v\f\r") == std::string::npos ; 
}

inline bool sstr::isdigit_(char c ) { return std::isdigit(static_cast<unsigned char>(c)) ; }
inline bool sstr::isalnum_(char c ) { return std::isalnum(static_cast<unsigned char>(c)) ; }
inline bool sstr::isupper_(char c ) { return std::isupper(static_cast<unsigned char>(c)) ; }
inline bool sstr::islower_(char c ) { return std::islower(static_cast<unsigned char>(c)) ; }


/**
sstr::ParseIntSpec
--------------------

+------+--------------+-------------+
| spec | value        | scale       |
| (in) | (out)        | (in/out)    |
+======+==============+=============+
|  "1" |  1           |             | 
| "h1" |  100         |    100      |
| "H1" |  100000      |    100000   |
| "K1" |  1000        |    1000     |
| "K2" |  2000        |    1000     | 
| "M1" |  1000000     |   1000000   |
| "M2" |  2000000     |   1000000   |
+------+--------------+-------------+

* "K" or "M" are prefixes to avoid lots of zeros 
* spec with prefix both uses the scales to multiply the value and sets the scale for subsequent
* spec without prefix is multiplied by the current scale 

+-----+-------------+
| pfx |   scale     |
+=====+=============+
|  h  |       100   | 
|  K  |     1,000   | 
|  H  |   100,000   |
|  M  | 1,000,000   |
+-----+-------------+

Examples::

    H1,2,3,4,5,6,7,8,9,10


**/

template<typename T>
inline T sstr::ParseIntSpec( const char* spec, T& scale ) // static 
{
    bool valid = spec != nullptr && strlen(spec) > 0 ; 
    if(!valid) return 0 ; 
    bool is_digit = isdigit_(spec[0]);  
    const char* e = is_digit ? spec : spec + 1 ; 
    T value = To<T>( e) ; 
    ParseScale<T>(spec, scale); 
    return value*scale  ; 
}

template<typename T>
inline T sstr::ParseInt( const char* spec) // static 
{
    T scale(1) ; 
    return ParseIntSpec<T>( spec, scale ); 
}



template<typename T>
inline void sstr::ParseScale( const char* spec, T& scale )
{
    bool is_digit = isdigit_(spec[0]);  
    if(!is_digit)
    {
        switch(spec[0])
        {
            case 'h': scale = 100     ; break ;  
            case 'K': scale = 1000    ; break ;  
            case 'H': scale = 100000   ; break ;  
            case 'M': scale = 1000000 ; break ;  
        }
    }
}


/**
sstr::ParseIntSpecList
------------------------

Parses delimited string into vector of ints, for examples

+---------------------+------------------------------------------------------+
|  spec               | values                                               | 
+=====================+======================================================+
|  "M1,2,3,4,5,K1,2"  | 1000000,2000000,3000000,4000000,5000000,1000,2000    |
|  "M1:5,K1:2"        | 1000000,2000000,3000000,4000000,5000000,1000,2000    |
+---------------------+------------------------------------------------------+


**/

template<typename T>
inline void sstr::ParseIntSpecList( std::vector<T>& values, const char* spec, char delim ) // static 
{
    values.clear(); 
    std::stringstream ss; 
    ss.str(spec)  ;
    std::string elem ;
    T scale = 1 ; 
    while (std::getline(ss, elem, delim)) 
    {
        const char* e = elem.c_str(); 
        const char* p = strstr(e, ":" ); 

        if( p == nullptr )
        {
            values.push_back(ParseIntSpec<T>( e, scale )); 
        }
        else
        {
            const char* q = isdigit_(e[0]) ? e : e + 1 ; 
            T i0, i1 ; 
            ParsePair<T>( q , i0, i1, ':' );
            ParseScale<T>( e, scale ); 
            for(T i=i0 ; i <= i1 ; i++) values.push_back(i*scale) ;      
        }
    }
}

template<typename T>
inline std::vector<T>* sstr::ParseIntSpecList( const char* spec, char delim )
{
    if(spec == nullptr) return nullptr ; 
    std::vector<T>* ls = new std::vector<T> ; 
    ParseIntSpecList<T>( *ls, spec, delim ); 
    return ls ; 
}

inline void sstr::truncated_copy( char* dst, const char* src, int dst_size ) 
{
    memset(dst, 0, dst_size); 

    int srclen = strlen(src) ;
    int num_char = std::min(dst_size, srclen); 
    memcpy(dst, src, num_char) ; 
}



/**
sstr::Extract
-----------------------

Extract integers from a string into a vector. 

The 2nd strtol endptr arg increments p beyond each group of integer digits

**/

inline void sstr::Extract( std::vector<long>& vals, const char* s )  // static
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

inline long sstr::ExtractLong( const char* s, long fallback ) // static
{
    std::vector<long> vals;
    Extract(vals, s); 
    return vals.size() == 1 ? vals[0] : fallback ; 
}




