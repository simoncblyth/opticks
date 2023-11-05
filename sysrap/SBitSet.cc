#include <cstring>
#include <sstream>
#include <iomanip>

#include "sstr.h"
#include "ssys.h"

#include "SBitSet.hh"
#include "SName.h"

std::string SBitSet::Brief( const SBitSet* elv, const SName* id )
{
    std::stringstream ss ; 
    ss << "SBitSet::Brief" ; 
    std::string str = ss.str(); 
    return str ; 
}


/**
SBitSet::Create
-----------------

1. reads spec from ekey envvar eg:"ELV" with fallback eg:"t"
2. 

**/


SBitSet* SBitSet::Create(unsigned num_bits, const char* ekey, const char* fallback )
{
    const char* spec = ssys::getenvvar(ekey, fallback) ;
    SBitSet* bs = Create(num_bits, spec); 
    if(bs) bs->set_label(ekey); 
    return bs ; 
}

SBitSet* SBitSet::Create(unsigned num_bits, const char* spec)
{
    SBitSet* bs = nullptr ; 
    if(spec)
    {
        bs = new SBitSet(num_bits); 
        bs->parse(spec);  
    }
    return bs ; 
}


/**
SBitSet::Parse
----------------

Interpret spec string into a array of bool indicating positions where bits are set 

When the first char of spec is '~' or 't' indicating complement all bits 
are first set *true* and the comma delimited bit positions provided are set to *false*. 

Otherwise without the complement token all bit positions are first set *false*
and the bit positions provided are set to *true*.

Examples with num_bits:8 for brevity:

* NB : think of the bitset as a sequence, not as a number 

+-------+--------------------+---------------------------------+
| spec  | bits (num_bits:8)  |  notes                          |
+=======+====================+=================================+
|  t    |  11111111          | num_bits all set                |
+-------+--------------------+---------------------------------+
|       |  00000000          | blank string spec               |
+-------+--------------------+---------------------------------+
|  t0   |  01111111          |                                 |
+-------+--------------------+---------------------------------+
|  0    |  10000000          |                                 | 
+-------+--------------------+---------------------------------+
|  1    |  01000000          |                                 |
+-------+--------------------+---------------------------------+
|  0,2  |  10100000          |                                 |
+-------+--------------------+---------------------------------+
|  t0,2 |  01011111          |                                 |
+-------+--------------------+---------------------------------+

* see SBitSetTest for more examples 

TODO: unify the spec approach used by ELV and EMM (EMM uses
a similar but different approach that will cause confusion)

**/

void SBitSet::Parse( unsigned num_bits, bool* bits,  const char* spec )
{
    bool with_complement = strlen(spec) > 0 && ( spec[0] == '~' || spec[0] == 't' ) ; // str_ starts with ~ or t 
    for(unsigned i=0 ; i < num_bits ; i++) bits[i] = with_complement ? true : false ; 

    bool with_colon = strlen(spec) >= 2 && spec[1] == ':' ;  

    int post_complement =  with_complement ? 1 : 0 ;  // offset to skip the complement first character                     
    int post_colon =  with_colon ? 1 : 0 ;  

    const char* spec_ = spec + post_complement + post_colon ; 

    std::vector<int> pos ; 
    char delim = ',' ; 
    sstr::split<int>( pos,  spec_ , delim );

    for(unsigned i=0 ; i < pos.size() ; i++)
    {
        int ipos = pos[i] ; 
        int upos_ = ipos < 0 ? ipos + num_bits : ipos ; 
        assert( upos_ > -1 ); 
        unsigned upos = upos_ ;  
        assert( upos < num_bits ); 

        bits[upos] = with_complement ? false : true ;    
    }
}     

std::string SBitSet::Desc( unsigned num_bits, const bool* bits, bool reverse )
{
    std::stringstream ss ; 
    ss << std::setw(4) << num_bits << " : " ;  
    for(unsigned i=0 ; i < num_bits ; i++)  ss << ( bits[reverse ? num_bits - 1 - i : i ] ? "1" : "0" ) ; 
    std::string s = ss.str(); 
    return s ; 
}

bool SBitSet::is_set(unsigned pos) const 
{
    assert( pos < num_bits ); 
    return bits[pos] ; 
}

unsigned SBitSet::count() const 
{
    unsigned num = 0 ; 
    for(unsigned i=0 ; i < num_bits ; i++ ) if(bits[i]) num += 1 ;  
    return num ; 
}

bool SBitSet::is_all_set() const { return all() ; }
bool SBitSet::all() const { return count() == num_bits ; }
bool SBitSet::any() const { return count() > 0  ; }
bool SBitSet::none() const { return count() == 0  ; }

/**
SBitSet::get_pos
-----------------

Append to *pos* vector bit indices that are set OR notset 
depending on *value* bool.  

**/
void SBitSet::get_pos( std::vector<unsigned>& pos, bool value) const 
{
    for(unsigned i=0 ; i < num_bits ; i++ ) if(bits[i] == value) pos.push_back(i) ; 
}

SBitSet::SBitSet( unsigned num_bits_ )
    :
    num_bits(num_bits_),
    bits(new bool[num_bits]),
    label(nullptr),
    spec(nullptr)
{
    set(false); 
}

void SBitSet::set_label(const char* label_) // eg ELV or EMM 
{
    label = strdup(label_); 
}
void SBitSet::set_spec( const char* spec_)  // eg t or t0 t1 t0,1,2
{
    spec = strdup(spec_); 
}


void SBitSet::set(bool all)
{
    for(unsigned i=0 ; i < num_bits ; i++ )  bits[i] = all ; 
}

void SBitSet::parse(const char* spec)
{
    Parse(num_bits, bits, spec); 
    set_spec(spec); 
}


SBitSet::~SBitSet()
{ 
    delete [] bits ; 
}

std::string SBitSet::desc() const 
{
    std::stringstream ss ; 
    ss 
        << std::setw(4) << ( label ? label : "-" ) 
        << std::setw(10) << ( spec ? spec : "-" ) 
        << Desc(num_bits, bits, false)
        ;

    std::string s = ss.str();  
    return s ; 
  
}

