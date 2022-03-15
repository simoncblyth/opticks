#include <cstring>
#include <sstream>
#include <iomanip>

#include "SBitSet.hh"
#include "SSys.hh"
#include "SStr.hh"



SBitSet* SBitSet::Create(unsigned num_bits, const char* ekey, const char* fallback )
{
    const char* spec = SSys::getenvvar(ekey, fallback) ;
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

**/

void SBitSet::Parse( unsigned num_bits, bool* bits,  const char* spec )
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

bool SBitSet::all() const { return count() == num_bits ; }
bool SBitSet::any() const { return count() > 0  ; }
bool SBitSet::none() const { return count() == 0  ; }

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

