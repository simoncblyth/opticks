#include "SStr.hh"
#include "PLOG.hh"

#include "scuda.h"

#include "CSGFoundry.h"
#include "CSGName.h"


const plog::Severity CSGName::LEVEL = PLOG::EnvLevel("CSGName", "DEBUG"); 

CSGName::CSGName( const CSGFoundry* foundry_ )
    :
    foundry(foundry_),
    name(foundry->meshname)
{
}

unsigned CSGName::getNumName() const
{
    return name.size(); 
}
const char* CSGName::getName(unsigned idx) const 
{
    return idx < name.size() ? name[idx].c_str() : nullptr ; 
}

/**
CSGName::getAbbr
------------------

Return the shortest string that still yields the same index 

**/

const char* CSGName::getAbbr(unsigned idx) const
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
        LOG(info) 
           << " idx " << idx
           << " idx0 " << idx0
           << " count " << count
           << " name " << name
           << " sname " << sname
           << " nj " << nj
           ;
    }


    unsigned max_count = 2 ;  // strict, but permit duplicated
    for(int j=0 ; j < nj ; j++) 
    {
        sname[nj-1-j] = '\0' ;   // progressive trimming from the right 
        count = 0 ; 
        int idx1 = findIndex(sname, count, max_count ); 

        if( idx == 0 ) 
           LOG(info) 
               << " j " << j  
               << " sname " << sname 
               << " idx1 " << idx1 
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
CSGName::getIndex
--------------------

Returns the index of the first listed name that exactly matches the query string.
A count of the number of matches is also provided.
Returns -1 if not found.

**/

int CSGName::getIndex(const char* query, unsigned& count) const 
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
CSGName::findIndex
--------------------

Returns the index of the first listed name that starts with the query string.
A count of the number of matches is also provided.

When max_count argument > -1  is provided, eg max_count=1 
the number of occurences of the match is required to be less than 
or equal to *max_count*.

Returns -1 if not found.

**/

int CSGName::findIndex(const char* starting, unsigned& count, int max_count ) const 
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

/**
CSGName::parseArg
-------------------

An arg of "ALL" is special cased yielding -1 otherwise parsing the string
as an integer is attempted. If the entire string does not parse as an integer 
or it matches the fallback "-1" then look for the string in the list of names. 
If a name starting with the arg is found the 0-based index is returned, 
otherwise -1 is returned.   

**/

const char* CSGName::parseArg_ALL = "ALL" ; 

int CSGName::parseArg(const char* arg, unsigned& count) const 
{
    count = 0 ; 
    int idx = -1 ; 

    bool is_all = strcmp( arg, parseArg_ALL) == 0 ? true : false ; 
    if(is_all)
    {
        count = 1 ; 
    }
    else
    {
        int fallback = -1 ; 
        int idx = ParseIntString(arg, fallback ) ; 
     
        if(idx == fallback)  
        {   
            idx = findIndex(arg, count);  
        }   
        else
        {   
            count = 1 ; 
        }
    }

    LOG(LEVEL) << " arg " << arg << " idx " << idx << " count " << count << " is_all " << is_all ; 
    return idx ; 
}

/**
CSGName::ParseIntString
-------------------------

If the entire arg can be parsed as an integer that is returned,  
otherwise the fallback integer is returned.

**/

int CSGName::ParseIntString(const char* arg, int fallback)  // static 
{
    char* end ;   
    char** endptr = &end ; 
    int base = 10 ;   
    unsigned long int uli = strtoul(arg, endptr, base); 
    bool end_points_to_terminator = end == arg + strlen(arg) ;   
    int result = int(uli) ; 
    return end_points_to_terminator ? result : fallback ;
}

/**
CSGName::parseMOI
-------------------

Parses MOI string into three integers:

midx
    mesh index
mord
    mesh ordinal 
iidx
    instance index


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

void CSGName::parseMOI(int& midx, int& mord, int& iidx, const char* moi) const 
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

    LOG(LEVEL) 
        << " moi " << moi 
        << " num_elem " << num_elem
        << " count " << count 
        << " midx " << midx 
        << " mord " << mord
        << " iidx " << iidx
        << " name.size " << name.size()
        ;


}


