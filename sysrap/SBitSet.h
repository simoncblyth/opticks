#pragma once
/**
SBitSet.h
=============

Used for example from CSGFoundry::Load to implement dynamic prim selection.

For example with ELV the num_bits_ would be the number of unique solid shapes in the geometry,
with is around 140 for JUNO.  So the bitset provides a way to represent a selection
over those shapes.

Note that at this level, there are no geometry specifics like names etc... its
just a collection of booleans.

**/

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <sstream>
#include <iomanip>

#include "sstr.h"
#include "ssys.h"
#include "SName.h"


struct SBitSet
{
    static std::string Brief( const SBitSet* elv, const SName* id );

    template<typename T>
    static T          Value( unsigned num_bits, const char* ekey, const char* fallback );

    template<typename T>
    static T          Value( unsigned num_bits, const char* spec );


    static SBitSet*   Create( unsigned num_bits, const char* ekey, const char* fallback );
    static SBitSet*   Create( unsigned num_bits, const char* spec);


    template<typename T>
    static  SBitSet* Create(T value);


    static int ParseSpec( unsigned num_bits, bool* bits      , const char* spec, bool dump );

    template<typename T>
    static void        ParseValue( unsigned num_bits, bool* bits      , T value );

    template<typename T>
    static bool IsSet( T value, int ibit );


    static std::string Desc(  unsigned num_bits, const bool* bits, bool reverse );

    template<typename T>
    static std::string DescValue( T val    );


    static constexpr const char* __level = "SBitSet__level" ;
    int         level ;
    unsigned    num_bits ;
    bool*       bits ;

    // metadata
    const char* label ;
    const char* spec ;


    SBitSet( unsigned num_bits);
    virtual ~SBitSet();

    void        set_label(const char* label);
    void        set_spec( const char* spec);

    void        set(bool all);
    void        parse_spec( const char* spec);



    template<typename T>
    void        parse_value( T value);

    bool        is_set(unsigned pos) const ;

    unsigned    count() const ;
    bool        is_all_set() const ;
    bool        all() const ;
    bool        any() const ;
    bool        none() const ;


    void get_pos( std::vector<unsigned>& pos, bool value ) const ;

    std::string desc() const ;

    template<typename T>
    T  value() const ;  // HMM: little or big endian option ?


    void serialize(std::vector<unsigned char>& buf) const ;
    static SBitSet* CreateFromBytes(const std::vector<unsigned char>& bytes);
    int compare(const SBitSet* other) const ; // zero when same bits

};


inline std::string SBitSet::Brief( const SBitSet* elv, const SName* id )
{
    std::stringstream ss ;
    ss << "SBitSet::Brief" ;
    std::string str = ss.str();
    return str ;
}



template<typename T>
inline T SBitSet::Value( unsigned num_bits, const char* ekey, const char* fallback )
{
    SBitSet* bs = Create(num_bits, ekey, fallback);
    T val = bs->value<T>() ;
    delete bs ;
    return val ;
}

template<typename T>
inline T SBitSet::Value( unsigned num_bits, const char* spec )
{
    SBitSet* bs = Create(num_bits, spec);
    T val = bs->value<T>() ;
    delete bs ;
    return val ;
}






/**
SBitSet::Create
-----------------

1. reads spec from ekey envvar eg:"ELV" with fallback eg:"t"
2.

**/


inline SBitSet* SBitSet::Create(unsigned num_bits, const char* ekey, const char* fallback )
{
    const char* spec = ssys::getenvvar(ekey, fallback) ;
    SBitSet* bs = Create(num_bits, spec );
    if(bs) bs->set_label(ekey);
    return bs ;
}

inline SBitSet* SBitSet::Create(unsigned num_bits, const char* spec )
{
    SBitSet* bs = nullptr ;
    if(spec)
    {
        bs = new SBitSet(num_bits);
        bs->parse_spec(spec);
    }
    return bs ;
}


template<typename T>
inline SBitSet* SBitSet::Create(T value)
{
    unsigned _num_bits = 8*sizeof(T);
    SBitSet* bs = new SBitSet(_num_bits);
    bs->parse_value(value);
    return bs ;
}







/**
SBitSet::ParseSpec
--------------------

Interpret spec string into a array of bool indicating positions where bits are set

with_complement:true
    when the first char of spec is '~' or 't' indicating that
    all bits are first set *true* and the comma delimited bit positions
    provided are set to *false*.

with_complement:false
    without complement token in first char all bit positions are first set *false*
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

inline int SBitSet::ParseSpec( unsigned num_bits, bool* bits,  const char* spec, bool dump )
{
    bool with_complement = strlen(spec) > 0 && ( spec[0] == '~' || spec[0] == 't' ) ; // str_ starts with ~ or t
    for(unsigned i=0 ; i < num_bits ; i++) bits[i] = with_complement ? true : false ;

    bool with_colon = strlen(spec) >= 2 && spec[1] == ':' ;

    int post_complement =  with_complement ? 1 : 0 ;  // offset to skip the complement first character
    int post_colon =  with_colon ? 1 : 0 ;
    const char* spec_ = spec + post_complement + post_colon ;

    // HUH: with_colon not used

    if(dump) std::cout
        << "[SBitSet::ParseSpec\n"
        << " dump " << ( dump ? "YES" : "NO " )
        << " num_bits " << num_bits << "\n"
        << " spec [" <<  ( spec  ? spec  : "-" ) << "]" << "\n"
        << " spec_[" <<  ( spec_ ? spec_ : "-" ) << "]" << "\n"
        << " with_complement " << (  with_complement ? "YES" : "NO " ) << "\n"
        << " with_colon " << (  with_colon ? "YES" : "NO " ) << "\n"
        << "\n"
        ;

    std::vector<int> pos ;
    char delim = ',' ;
    sstr::split<int>( pos,  spec_ , delim );

    int rc = 0;

    for(unsigned i=0 ; i < pos.size() ; i++)
    {
        int ipos = pos[i] ;
        int upos_ = ipos < 0 ? ipos + num_bits : ipos ;

        assert( upos_ > -1 );

        unsigned upos = upos_ ;
        bool in_range = upos < num_bits ;
        if(!in_range) std::cerr
            << "-SBitSet::ParseSpec"
            << " i " << i
            << " pos.size " << pos.size()
            << " ipos " << ipos
            << " upos_ " << upos_
            << " upos " << upos
            << " num_bits " << num_bits
            << " in_range " << ( in_range ? "YES" : "NO " )
            << "\n"
            ;


        if(!in_range) rc += 1 ;
        //assert( expect );

        if(upos < num_bits) bits[upos] = with_complement ? false : true ;
    }

    if(dump) std::cout
        << " rc " << rc
        << "]SBitSet::ParseSpec\n"
        ;

    return rc;

}





template<typename T>
inline void SBitSet::ParseValue( unsigned num_bits, bool* bits,  T value )
{
    assert( sizeof(T)*8 == num_bits );
    for(unsigned i=0 ; i < num_bits ; i++ ) bits[i] = IsSet(value, i) ;
}

template<typename T>
inline bool SBitSet::IsSet( T value, int _ibit )  // static
{
    unsigned num_bits = sizeof(T)*8 ;
    T ibit = _ibit < 0 ? _ibit + num_bits : _ibit ;
    T mask = 0x1 << ibit ;
    bool is_set = ( value & mask ) != 0 ;
    return is_set ;
}



inline std::string SBitSet::Desc( unsigned num_bits, const bool* bits, bool reverse )
{
    std::stringstream ss ;
    ss << std::setw(4) << num_bits << " : " ;
    for(unsigned i=0 ; i < num_bits ; i++)  ss << ( bits[reverse ? num_bits - 1 - i : i ] ? "1" : "0" ) ;
    std::string str = ss.str();
    return str ;
}

template<typename T>
inline std::string SBitSet::DescValue( T val  )
{
    std::stringstream ss ;
    ss << "bs.0x" << std::hex << val << std::dec ;
    std::string str = ss.str();
    return str ;
}



inline bool SBitSet::is_set(unsigned pos) const
{
    assert( pos < num_bits );
    return bits[pos] ;
}

inline unsigned SBitSet::count() const
{
    unsigned num = 0 ;
    for(unsigned i=0 ; i < num_bits ; i++ ) if(bits[i]) num += 1 ;
    return num ;
}

inline bool SBitSet::is_all_set() const { return all() ; }
inline bool SBitSet::all() const { return count() == num_bits ; }
inline bool SBitSet::any() const { return count() > 0  ; }
inline bool SBitSet::none() const { return count() == 0  ; }

/**
SBitSet::get_pos
-----------------

Append to *pos* vector bit indices that are set OR notset
depending on *value* bool.

**/
inline void SBitSet::get_pos( std::vector<unsigned>& pos, bool value) const
{
    for(unsigned i=0 ; i < num_bits ; i++ ) if(bits[i] == value) pos.push_back(i) ;
}

inline SBitSet::SBitSet( unsigned num_bits_ )
    :
    level(ssys::getenvint(__level,0)),
    num_bits(num_bits_),
    bits(new bool[num_bits]),
    label(nullptr),
    spec(nullptr)
{
    set(false);
}

inline void SBitSet::set_label(const char* label_) // eg ELV or EMM
{
    label = strdup(label_);
}
inline void SBitSet::set_spec( const char* spec_)  // eg t or t0 t1 t0,1,2
{
    spec = strdup(spec_);
}


inline void SBitSet::set(bool all)
{
    for(unsigned i=0 ; i < num_bits ; i++ )  bits[i] = all ;
}


inline void SBitSet::parse_spec(const char* spec)
{
    bool dump = level > 0 ;
    ParseSpec(num_bits, bits, spec, dump);
    set_spec(spec);
}

template<typename T>
inline void SBitSet::parse_value(T value)
{
    ParseValue(num_bits, bits, value);
}




inline SBitSet::~SBitSet()
{
    delete [] bits ;
}

inline std::string SBitSet::desc() const
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


template<typename T>
inline T SBitSet::value() const   // HMM: little or big endian option ?
{
    T val = 0 ;
    for(unsigned i=0 ; i < num_bits ; i++ )
    {
       if(bits[i]) val |= ( 0x1 << i )  ;
    }
    return val ;
}


inline void SBitSet::serialize(std::vector<unsigned char>& bytes) const
{
    int num_bytes = (num_bits + 7)/8 ;
    bytes.resize(num_bytes, 0);
    for (size_t i = 0; i < num_bits; ++i) if(bits[i]) bytes[i/8] |= (1 << (i % 8));
}

inline SBitSet* SBitSet::CreateFromBytes(const std::vector<unsigned char>& bytes) // static
{
    int num_bits = 8*bytes.size() ;
    SBitSet* bs = new SBitSet(num_bits);
    for (int i = 0; i < num_bits; ++i)
    {
        unsigned mask = 0x1 << (i % 8) ;
        unsigned char byte = bytes[i/8] ;
        bs->bits[i] = byte & mask ;
    }
    return bs ;
}

inline int SBitSet::compare(const SBitSet* other) const
{
    if(!other) return -1 ;
    if(num_bits != other->num_bits) return -2 ;
    int diff = 0 ;
    for(unsigned i=0 ; i < num_bits ; i++) diff += ( bits[i] == other->bits[i] ? 0 : 1 ) ;
    return diff ;
}


