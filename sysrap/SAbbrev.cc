#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>

#include "SASCII.hh"
#include "SAbbrev.hh"


SAbbrev::SAbbrev( const std::vector<std::string>& names_ ) 
    :
    names(names_)
{
    init();
}

bool SAbbrev::isFree(const std::string& ab) const 
{
    return std::find( abbrev.begin(), abbrev.end(), ab ) == abbrev.end() ; 
}

void SAbbrev::init()
{
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        SASCII n(names[i].c_str());  

        std::string ab ; 

        if( n.upper > 1 )
        { 
            ab = n.getFirstUpper(2) ; 
        }
        else 
        {
            ab = n.getFirst(2) ; 
        }

        if(!isFree(ab))
        {
            ab = n.getFirstLast(); 
        } 

        assert( isFree(ab) && "failed to abbreviate "); 
        abbrev.push_back(ab) ;  
    }
}

void SAbbrev::dump() const 
{
    for(unsigned i=0 ; i < names.size() ; i++)
    {
         std::cout 
             << std::setw(30) << names[i]
             << " : " 
             << std::setw(2) << abbrev[i]
             << std::endl 
             ;
    }
}




