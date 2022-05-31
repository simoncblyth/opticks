#pragma once

/**
SName.h  : formerly CSG/CSGName.h
=====================================

Identity machinery using the foundry vector of meshnames (aka solid names) 

**/

#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "SStr.hh"


enum { SName_EXACT, SName_START, SName_CONTAIN } ; 

struct SName
{
    static constexpr const char* EXACT = "EXACT" ; 
    static constexpr const char* START = "START" ; 
    static constexpr const char* CONTAIN = "CONTAIN" ; 

    static unsigned QType(char qt); 
    static const char* QLabel(unsigned qtype); 
    static const char* QTypeLabel(char qt); 
    static bool Match( const char* n, const char* q, unsigned qtype ); 
    static SName* Load(const char* path); 

    static constexpr const char* parseArg_ALL = "ALL" ; 
    static const bool dump = false ; 
    static int ParseIntString(const char* arg, int fallback=-1);
    static void ParseSOPR(int& solidIdx, int& primIdxRel, const char* sopr ); 


    const std::vector<std::string>& name ; 

    SName(const std::vector<std::string>& name );  

    std::string desc() const ; 
    std::string detail() const ; 

    unsigned getNumName() const;
    const char* getName(unsigned idx) const ;
    const char* getAbbr(unsigned idx) const ;

    int getIndex( const char* name    , unsigned& count) const ;
    int findIndex(const char* starting, unsigned& count, int max_count=-1) const ;
    int findIndex(const char* starting) const ;
    void findIndicesFromNames(std::vector<unsigned>& idxs, const std::vector<std::string>& qq ) const ; 

    bool hasName(  const char* q ) const ; 
    bool hasNames( const char* qq, char delim=',' ) const ; 
    bool hasNames( const std::vector<std::string>& qq ) const ; 


    void findIndices(std::vector<unsigned>& idxs, const char* query, char qt='S' ) const ; 
    std::string descIndices(const std::vector<unsigned>& idxs) const ; 

    static const char* ELVString(const std::vector<unsigned>& idxs, const char* prefix="t" ); 

    const char* get_ELV_fromNames( const char* names, char delim=',' ) const ;  
    const char* get_ELV_fromNames( const std::vector<std::string>& names ) const ; 
    const char* get_ELV_fromSearch( const char* names_containing="_virtual0x" ) const;   


    int parseArg(const char* arg, unsigned& count ) const ;
    void parseMOI(int& midx, int& mord, int& iidx, const char* moi) const ;

}; 

inline SName* SName::Load(const char* path)
{
    typedef std::vector<std::string> VS ; 
    VS* names = new VS ; 

    std::ifstream ifs(path);
    std::string line;
    while(std::getline(ifs, line)) names->push_back(line) ; 

    SName* id = new SName(*names) ; 
    return id ;
}


inline SName::SName( const std::vector<std::string>& name_ )
    :
    name(name_)
{
}

inline std::string SName::desc() const 
{
    unsigned num_name = getNumName() ; 
    std::stringstream ss ; 
    ss << "SName::desc numName " << num_name << " name[0] " << getName(0) << " name[-1] " <<  getName(num_name-1 ) ; 
    std::string s = ss.str(); 
    return s ; 
}

inline std::string SName::detail() const 
{
    unsigned num_name = getNumName() ; 
    std::stringstream ss ; 
    ss << "SName::detail num_name " << num_name << std::endl ;
    for(unsigned i=0 ; i < num_name ; i++) ss << getName(i) << std::endl ;  
    std::string s = ss.str(); 
    return s ; 
}


inline unsigned SName::getNumName() const
{
    return name.size(); 
}
inline const char* SName::getName(unsigned idx) const 
{
    return idx < name.size() ? name[idx].c_str() : nullptr ; 
}

/**
SName::getAbbr
------------------

Return the shortest string that still yields the same index 

**/

inline const char* SName::getAbbr(unsigned idx) const
{
    const char* name = getName(idx); 
   
    unsigned count = 0 ; 
    int idx0 = getIndex(name, count) ; 

    if( idx0 != int(idx) )  return nullptr ;  // happens for 2nd of duplicated
    // count is 2 for the first of duplicated 

    char* sname = strdup(name); 
    int nj = int(strlen(sname)); 

    if( idx == 0 )
    {
        std::cout
           << " idx " << idx
           << " idx0 " << idx0
           << " count " << count
           << " name " << name
           << " sname " << sname
           << " nj " << nj
           << std::endl 
           ;
    }


    unsigned max_count = 2 ;  // strict, but permit duplicated
    for(int j=0 ; j < nj ; j++) 
    {
        sname[nj-1-j] = '\0' ;   // progressive trimming from the right 
        count = 0 ; 
        int idx1 = findIndex(sname, count, max_count ); 

        if( idx == 0 ) 
           std::cout
               << " j " << j  
               << " sname " << sname 
               << " idx1 " << idx1 
               << std::endl 
               ;


        if(idx1 != int(idx) )   
        {
            sname[nj-1-j] = name[nj-1-j] ; // repair the string  
            break ;  
        }     
    }
    return sname ;     
}






/**
SName::getIndex
--------------------

Returns the index of the first listed name that exactly matches the query string.
A count of the number of matches is also provided.
Returns -1 if not found.

NB NP::get_name_index does the same as this, it can be simpler to use that 
method when an array is being updated

**/

inline int SName::getIndex(const char* query, unsigned& count) const 
{
    int result(-1); 
    count = 0 ; 
    for(unsigned i=0 ; i < name.size() ; i++)
    {   
        const std::string& k = name[i] ;
        if(strcmp(k.c_str(), query) == 0 )
        {
            if(count == 0) result = i ; 
            count += 1 ;  
        }  
    }
    return result ; 
}


/**
SName::findIndex
--------------------

Returns the index of the first listed name that starts with the query string.
A count of the number of matches is also provided.

Start strings are used to allow names with pointer suffixes such as
the below to be found without including the pointer suffix in the query string::

   HamamatsuR12860sMask_virtual0x5f50520
   HamamatsuR12860sMask_virtual0x

It is recommended to make the point by using query strings ending in 0x 

When max_count argument > -1  is provided, eg max_count=1 
the number of occurences of the match is required to be less than 
or equal to *max_count*.

Returns -1 if not found.

**/

inline int SName::findIndex(const char* starting, unsigned& count, int max_count ) const 
{  
    int result(-1); 
    count = 0 ; 
    for(unsigned i=0 ; i < name.size() ; i++)
    {   
        const std::string& k = name[i] ;
        if( SStr::StartsWith( k.c_str(), starting ))  
        {   
            if(count == 0) result = i ; 
            count += 1 ;  
        }   
    }   
    bool count_ok = max_count == -1 || count <= unsigned(max_count) ; 
    return count_ok ? result : -1 ;   
}

inline int SName::findIndex(const char* starting) const 
{
    unsigned count = 0 ;  
    int max_count = -1 ; 
    int idx = findIndex(starting, count, max_count); 
    return idx ; 
}

inline void SName::findIndicesFromNames(std::vector<unsigned>& idxs, const std::vector<std::string>& qq ) const 
{
    for(unsigned i=0 ; i < qq.size() ; i++)
    {   
        const char* q = qq[i].c_str(); 
        int idx = findIndex(q) ;  
        bool found = idx > -1 ; 
        if(!found) std::cerr << "SName::findIndicesfromNames FAILED to find q [" << q << "]" << std::endl ; 
        assert(found);  
        idxs.push_back(idx) ;  
    }
}

inline bool SName::hasName(  const char* q ) const 
{
    int idx = findIndex(q); 
    bool has = idx > -1 ; 
    return has ; 
}
inline bool SName::hasNames( const char* qq_, char delim ) const 
{
    std::vector<std::string> qq; 
    SStr::Split(qq_, delim, qq); 
    return hasNames(qq); 
}
inline bool SName::hasNames( const std::vector<std::string>& qq ) const 
{
    std::vector<unsigned> idxs ; 
    findIndicesFromNames(idxs, qq); 
    bool has_all = qq.size() == idxs.size() ; 
    return has_all ; 
}







inline const char* SName::QLabel(unsigned qtype)  // static
{
    const char* s = nullptr ; 
    switch(qtype)
    {
       case SName_EXACT :   s = EXACT ; break ;  
       case SName_START :   s = START ; break ;  
       case SName_CONTAIN : s = CONTAIN ; break ;  
    }
    return s ; 
}

inline unsigned SName::QType(char qt)  // static
{
    unsigned qtype = SName_EXACT ; 
    switch(qt) 
    {
        case 'E': qtype = SName_EXACT   ; break ; 
        case 'S': qtype = SName_START   ; break ; 
        case 'C': qtype = SName_CONTAIN ; break ; 
    }
    return qtype ; 
}
inline const char* SName::QTypeLabel(char qt) // static 
{
    unsigned qtype = QType(qt); 
    return QLabel(qtype); 
}


inline bool SName::Match( const char* n, const char* q, unsigned qtype ) // static
{
    bool match = false ; 
    switch( qtype )
    {
        case SName_EXACT:   match = strcmp(n,q) == 0       ; break ;  // n exactly matches q string  
        case SName_START:   match = SStr::StartsWith(n,q)  ; break ;  // n starts with q string  
        case SName_CONTAIN: match = SStr::Contains(n,q)    ; break ;  // n contains the q string
    }
    return match ; 
}

inline void SName::findIndices(std::vector<unsigned>& idxs, const char* q, char qt ) const 
{
    unsigned qtype = QType(qt); 
    for(unsigned i=0 ; i < name.size() ; i++)
    {   
        const char* n = name[i].c_str() ;
        if(Match(n,q,qtype)) idxs.push_back(i) ;  
    }
}


inline const char* SName::get_ELV_fromNames( const char* names_, char delim ) const 
{
    std::vector<std::string> names ; 
    SStr::Split(names_, delim, names); 
    return get_ELV_fromNames( names); 
}
inline const char* SName::get_ELV_fromNames( const std::vector<std::string>& qq ) const 
{
    std::vector<unsigned> idxs ; 
    findIndicesFromNames(idxs, qq); 
    assert( qq.size() == idxs.size() ); 
    const char* prefix = "t" ; 
    return ELVString(idxs, prefix); 
}
inline const char* SName::get_ELV_fromSearch( const char* names_containing ) const 
{  
    std::vector<unsigned> idxs ; 
    findIndices(idxs, names_containing, 'C' ); 
    const char* prefix = "t" ; 
    return ELVString(idxs, prefix); 
}  








inline std::string SName::descIndices(const std::vector<unsigned>& idxs) const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < idxs.size() ; i++)
    {
        unsigned idx = idxs[i] ; 
        ss << std::setw(4) << idx << " : " << name[idx] << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
}

inline const char* SName::ELVString(const std::vector<unsigned>& idxs, const char* prefix ) // static
{
    unsigned num_idx = idxs.size() ; 
    std::stringstream ss ; 
    ss << prefix ; 
    for(unsigned i=0 ; i < num_idx ; i++) ss << idxs[i] <<  ( i < num_idx - 1 ? "," : "" ) ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}


/**
SName::parseArg
-------------------

An arg of "ALL" is special cased yielding -1 otherwise parsing the string
as an integer is attempted. If the entire string does not parse as an integer 
or it matches the fallback "-1" then look for the string in the list of names. 
If a name starting with the arg is found the 0-based index is returned, 
otherwise -1 is returned.   

**/


inline int SName::parseArg(const char* arg, unsigned& count) const 
{
    count = 0 ; 

    int fallback = -1 ; 
    int idx = fallback ; 

    bool is_all = strcmp( arg, parseArg_ALL) == 0 ? true : false ; 
    if(is_all)
    {
        count = 1 ; 
    }
    else
    {
        idx = ParseIntString(arg, fallback ) ; 
        if(idx == fallback)  
        {   
            idx = findIndex(arg, count);  
        }   
        else
        {   
            count = 1 ; 
        }
    }

    if(dump) std::cout << " arg " << arg << " idx " << idx << " count " << count << " is_all " << is_all << std::endl ; 
    return idx ; 
}

/**
SName::ParseIntString
-------------------------

If the entire arg can be parsed as an integer that is returned,  
otherwise the fallback integer is returned.

**/

inline int SName::ParseIntString(const char* arg, int fallback)  // static 
{
    char* end ;   
    char** endptr = &end ; 
    int base = 10 ;   
    unsigned long int uli = strtoul(arg, endptr, base); 
    bool end_points_to_terminator = end == arg + strlen(arg) ;   
    int result = int(uli) ; 
    int ret = end_points_to_terminator ? result : fallback ;

    if(dump) std::cout  
         << " arg [" << arg << "] " 
         << " uli " << uli 
         << " end_points_to_terminator " << end_points_to_terminator
         << " result " << result 
         << " ret " << ret 
         << std::endl 
         ;

    return ret ;  
}


/**
SName::ParseSOPR
--------------------
**/

inline void SName::ParseSOPR(int& solidIdx, int& primIdxRel, const char* sopr_ ) // static
{
    const char* sopr = SStr::ReplaceChars(sopr_, "_", ':'); 

    std::stringstream ss; 
    ss.str(sopr)  ;
    std::string s;
    char delim = ':' ; 
    std::vector<std::string> elem ; 
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 

    unsigned num_elem = elem.size(); 
    
    solidIdx = num_elem > 0 ? ParseIntString( elem[0].c_str() ) : 0 ; 
    primIdxRel = num_elem > 1 ? ParseIntString( elem[1].c_str() ) : 0 ; 

    if(dump) std::cout 
        << " sopr_ " << sopr_ 
        << " sopr " << sopr 
        << " solidIdx " << solidIdx 
        << " primIdxRel " << primIdxRel 
        << std::endl 
        ; 
}



/**
SName::parseMOI
-------------------

Used from CSGFoundry::parseMOI


Parses MOI string into three integers:

midx
    mesh index
mord
    mesh ordinal 
iidx
    instance index
    (not the global instance index)


MOI are strings delimited by colons of form::

    sWorld:0:0 
    sWorld:0      # skipped integers default to zero 
    sWorld        # skipped integers default to zero 

    0:0:0
    0:0
    0

The first element of the string can be a string such as "sWorld" or an integer, 
subsequent elements are expected to be integers. 

**/

inline void SName::parseMOI(int& midx, int& mord, int& iidx, const char* moi ) const 
{
    std::stringstream ss; 
    ss.str(moi)  ;
    std::string s;
    char delim = ':' ; 
    std::vector<std::string> elem ; 
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 

    unsigned num_elem = elem.size(); 
    
    unsigned count = 0 ; 
    midx = num_elem > 0 ? parseArg( elem[0].c_str(), count) : 0 ;  
    mord = num_elem > 1 ? ParseIntString( elem[1].c_str() ) : 0 ; 
    iidx = num_elem > 2 ? ParseIntString( elem[2].c_str() ) : 0 ; 

    if(dump) std::cout
        << " moi " << moi 
        << " num_elem " << num_elem
        << " count " << count 
        << " midx " << midx 
        << " mord " << mord
        << " iidx " << iidx
        << " name.size " << name.size()
        << std::endl 
        ;
}



