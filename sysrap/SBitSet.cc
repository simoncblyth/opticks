#include <sstream>
#include "SBitSet.hh"
#include "SStr.hh"

/**
SBitSet::Parse
----------------

Interpret spec string into a array of bool indicating positions where bits are set 

When the first char of spec is '~' or 't' indicating complement all bits 
are first set *true* and the comma delimited bit positions provided are set to *false*. 

Otherwise without the complement token all bit positions are first set *false*
and the bit positions provided are set to *true*.

**/

void SBitSet::Parse( bool* bits, unsigned num_bits,  const char* spec )
{
    bool complement = strlen(spec) > 0 && ( spec[0] == '~' || spec[0] == 't' ) ;     // str_ starts with ~ or t 
    int postcomp =  complement ? 1 : 0 ;                                        // offset to skip the complement first character                     
    const char* spec_ = spec + postcomp ; 

    std::vector<int> pos ; 
    char delim = ',' ; 
    SStr::ISplit( spec_, pos, delim );  

    for(unsigned i=0 ; i < num_bits ; i++) bits[i] = complement ? true : false ; 

    for(unsigned i=0 ; i < pos.size() ; i++)
    {
        int ipos = pos[i] ; 
        int upos_ = ipos < 0 ? ipos + num_bits : ipos ; 
        assert( upos_ > -1 ); 
        unsigned upos = upos_ ;  
        assert( upos < num_bits ); 

        bits[upos] = complement ? false : true ;    
    }
}     


std::string SBitSet::Desc( bool* bits, unsigned num_bits, bool reverse )
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_bits ; i++)  ss << ( bits[reverse ? num_bits - 1 - i : i ] ? "1" : "0" ) ; 
    std::string s = ss.str(); 
    return s ; 
}


