// om-;TEST=SPackTest om-t 

#include <vector>
#include <cassert>
#include <csignal>
#include <iomanip>
#include "SPack.hh"
#include "OPTICKS_LOG.hh"

void test_Encode_Decode_unsigned()
{
    unsigned x = 0x00 ;   
    unsigned y = 0xaa ;   
    unsigned z = 0xbb ;   
    unsigned w = 0xff ;

    unsigned value_expect = 0xffbbaa00 ; 
    unsigned value = SPack::Encode(x,y,z,w); 

    LOG(info) 
        << " value " << std::hex << value 
        << " value_expect " << std::hex << value_expect
        ;

    assert( value == value_expect ); 

    unsigned x_ ; 
    unsigned y_ ; 
    unsigned z_ ; 
    unsigned w_ ; 

    SPack::Decode( value, x_, y_, z_, w_ ); 

    assert( x == x_ );   
    assert( y == y_ );   
    assert( z == z_ );   
    assert( w == w_ );   
}


void test_Encode()
{
    unsigned char nu = 10 ; 
    unsigned char nv = 10 ; 
    unsigned char nw =  4 ; 

    unsigned char u = nu - 1 ; 
    unsigned char v = nv - 1  ; 
    unsigned char w = nw - 1 ; 

    unsigned int packed = SPack::Encode(u,v,w,0); 
    LOG(info) 
        << " u " << u 
        << " v " << v 
        << " w " << w 
        << " packed " << packed 
        ;

}


void test_Encode_Decode()
{
    unsigned char x = 1 ; 
    unsigned char y = 128 ; 
    unsigned char z = 255 ; 
    unsigned char w = 128 ; 

    unsigned int value = SPack::Encode(x,y,z,w); 
    LOG(info) << " value " << value  ; 

    unsigned char x2, y2, z2, w2 ; 
    SPack::Decode( value, x2, y2, z2, w2 ); 

    assert( x == x2 ); 
    assert( y == y2 ); 
    assert( z == z2 ); 
    assert( w == w2 ); 
}


void test_Encode_Decode_ptr()
{
    unsigned char a[4] ; 
    a[0] = 1 ; 
    a[1] = 128 ; 
    a[2] = 255 ; 
    a[3] = 128 ; 

    unsigned int value = SPack::Encode(a, 4); 
    LOG(info) << " value " << value  ; 

    unsigned char b[4] ; 
    SPack::Decode( value, b, 4 ); 

    assert( a[0] == b[0] ); 
    assert( a[1] == b[1] ); 
    assert( a[2] == b[2] ); 
    assert( a[3] == b[3] ); 
}


void test_Encode13_Decode13()
{
    LOG(info); 

    unsigned char c  = 0xff ; 
    unsigned int ccc   = 0xffffff ; 
    unsigned expect  = 0xffffffff ; 

    unsigned value = SPack::Encode13( c, ccc );  
    bool value_expect = value == expect ; 
    assert( value_expect ); 
    if(!value_expect) std::raise(SIGINT); 

    unsigned char c2 ; 
    unsigned int  ccc2 ; 
    SPack::Decode13( value, c2, ccc2 ); 
    assert( c == c2 ); 
    assert( ccc == ccc2 ); 
}


void test_Encode22_Decode22()
{
    LOG(info); 

    unsigned a0 = 0xdead ; 
    unsigned b0 = 0xbeef ; 
    unsigned expect  = 0xdeadbeef ; 

    unsigned value = SPack::Encode22( a0, b0 );  
    bool value_expect = value == expect ; 
    assert( value_expect ); 
    if(!value_expect) std::raise(SIGINT); 


    unsigned a1 ; 
    unsigned b1 ; 
    SPack::Decode22( value, a1, b1 ); 

    bool decode22_expect = a0 == a1 && b0 == b1 ;
    assert( decode22_expect ); 
    if(!decode22_expect) std::raise(SIGINT);

    unsigned a2 = SPack::Decode22a( value ); 
    unsigned b2 = SPack::Decode22b( value ); 

    bool decode22a_expect = a0 == a2 && b0 == b2 ;
    assert( decode22a_expect ); 
    if(!decode22a_expect) std::raise(SIGINT);



}


void test_Encode22hilo_Decode22hilo(int a0, int b0, bool dump)
{
    unsigned packed = SPack::Encode22hilo( a0, b0 );  

    int a1 ; 
    int b1 ; 
    SPack::Decode22hilo(packed, a1, b1 ); 

    int a2 = SPack::Decode22hi(packed); 
    int b2 = SPack::Decode22lo(packed); 

    if(dump)
    {
        std::cout 
            << std::hex
            << " pk " << std::setw(8) << packed
            << "    "
            << " a0 " << std::setw(8) << a0
            << " a1 " << std::setw(8) << a1
            << " a2 " << std::setw(8) << a2
            << "    "
            << " b0 " << std::setw(8) << b0
            << " b1 " << std::setw(8) << b1
            << " b2 " << std::setw(8) << b2
            << std::endl
            ;
    }

    assert( a0 == a1 ); 
    assert( b0 == b1 ); 
    assert( a0 == a2 ); 
    assert( b0 == b2 ); 
}

void test_Encode22hilo_Decode22hilo()
{
    LOG(info); 
    int i0 = -0x8000 ; 
    int i1 =  0x7fff ; 
    int s = 10 ; 
    bool dump = false ; 

    for(int i=i0 ; i <= i1 ; i+=s ) for(int j=i0 ; j <= i1 ; j+=s ) test_Encode22hilo_Decode22hilo(i,j,dump); 

    typedef std::vector<std::pair<int,int>> VII ; 
    VII hilo = { 
                  { 0x0000, -0x0000},
                  { 0x0001, -0x0001},
                  { 0x0002, -0x0002},
                  { 0x0003, -0x0003},
                  { 0x0004, -0x0004},
                  { 0x0005, -0x0005},
                  { 0x0006, -0x0006},
                  { 0x0007, -0x0007},
                  { 0x0008, -0x0008},
               }; 

    for(VII::const_iterator it=hilo.begin() ; it != hilo.end() ; it++)
        test_Encode22hilo_Decode22hilo(  it->first,  it->second, true); 

    for(VII::const_iterator it=hilo.begin() ; it != hilo.end() ; it++)
        test_Encode22hilo_Decode22hilo( -it->first, -it->second, true); 

}



void test_int_as_float()
{
    int i0 = -420042 ;  
    float f0 = SPack::int_as_float( i0 ); 
    int i1 = SPack::int_from_float( f0 ); 
    assert( i0 == i1 ); 
    LOG(info) << " i0 " << i0 << " f0 " << f0 << " i1 " << i1 << " (NaN is expected) " ; 
}

void test_uint_as_float()
{
    unsigned u0 = 420042 ;  
    float f0 = SPack::uint_as_float( u0 ); 
    unsigned u1 = SPack::uint_from_float( f0 ); 
    assert( u0 == u1 ); 
    LOG(info) << " u0 " << u0 << " f0 " << f0 << " u1 " << u1 ; 
}


void test_IsLittleEndian()
{
    const char* LE = "LITTLE_ENDIAN : least significant byte at smaller memory address " ; 
    const char* BE = "BIG_ENDIAN    : most significant byte at smaller memory address " ; 
    LOG(info) << ( SPack::IsLittleEndian() ? LE : BE  ) ; 
}


void test_unsigned_as_int(int boundary, unsigned sensorIndex, bool dump)
{
    // LOG(info) << " boundary " << boundary ; 

    //unsigned packed = ( boundary << 16 | sensorIndex << 0 ); 
    //     simple packing like this doesnt work with signed ints, 
    //     must control the masking first otherwise the bits from eg -1:0xffffffff leak 

    unsigned packed = ((boundary & 0xffff) << 16 ) | ((sensorIndex & 0xffff) << 0 ) ;  
    unsigned hi = ( packed & 0xffff0000 ) >> 16 ;
    unsigned lo = ( packed & 0x0000ffff ) >>  0 ;

    // int hi_s = hi <= 0x7fff ? hi : hi - 0x10000 ;    // twos complement
    int hi_s = SPack::unsigned_as_int<16>(hi); 

    bool expect = hi_s == boundary && lo == sensorIndex ; 

    if(!expect || dump)
    std::cout 
        << " boundary " << std::setw(10) << std::dec << boundary 
        << " sensorIndex(hex) " << std::hex << sensorIndex
        << " packed(hex) " << std::hex << packed 
        << " hi(hex) " << std::hex << hi 
        << " lo(hex) " << std::hex << lo 
        << " hi_s " << std::dec << hi_s 
        << std::endl 
        ;

    assert(expect) ; 
}

void test_unsigned_as_int()
{
    //int boundary = -1 ; // 0xffff ;   // signed int that can easily fit into 16 bits 
    unsigned sensorIndex = 0xbeef ;   // unsigned int that can easily fit into 16 bits 
    unsigned signed_max_16 = (0x1 << (16 - 1)) - 1  ;   // 0x7fff  

    test_unsigned_as_int( -1                 , sensorIndex, true ); 
    test_unsigned_as_int( -(signed_max_16+1) , sensorIndex, true );
    test_unsigned_as_int(   signed_max_16    , sensorIndex, true );

    int boundary0 = -(signed_max_16+1) ; 
    int boundary1 = signed_max_16 ; 

    LOG(info) 
        << " boundary0 " << boundary0
        << " boundary1 " << boundary1
        ;

    for(int boundary=boundary0 ; boundary <= boundary1  ; boundary++)
        test_unsigned_as_int(boundary, sensorIndex, false); 

    //test_unsigned_as_int( -(signed_max_16+2), sensorIndex, false  );
    //test_unsigned_as_int(   (signed_max_16+1), sensorIndex, false  );

}


/**
test_unsigned_as_int_16
------------------------

This demonstrates that the union trick and twos-complement 
reinterpretation give the same result : although note that 
must use a union with elements of the appropriate number of bits.

**/

void test_unsigned_as_int_16(unsigned value)
{
    int v16_0 = SPack::unsigned_as_int<16>(value); 
    int v16_1 = SPack::unsigned_as_int_16( value ); 

    bool expect = v16_0 == v16_1 ; 
    bool dump = false ; 

    if(!expect || dump)
    {
        std::cout 
            << " v16_0 " << v16_0 
            << " v16_1 " << v16_1 
            << ( expect ? " " : " NOT-EXPECT " )
            << std::endl 
            ; 
    } 
    assert( expect ); 
}

void test_unsigned_as_int_16()
{
    unsigned value0 = 0 ; 
    unsigned value1 = ( 0x1 << 16 ) - 1 ; // 0xffff   
    for(unsigned value=value0 ; value <= value1  ; value++) test_unsigned_as_int_16(value); 
}

/**

OptiX_700/SDK/optixWhitted/helpers.h 

138 #define float3_as_args(u) \
139     reinterpret_cast<uint32_t&>((u).x), \
140     reinterpret_cast<uint32_t&>((u).y), \
141     reinterpret_cast<uint32_t&>((u).z)

**/


union uif_t {
   unsigned u ; 
   int      i ; 
   float    f ; 
};  


void dummy_optixTrace(unsigned& p0 )
{
    p0 += 100u ; 
}



#ifdef DEREFERENCING_TYPE_PUNNED_POINTER
void test_reinterpret_cast()
{
    unsigned u = 42 ; 
    uif_t uif ; 
    uif.u = u  ; 

    //unsigned u2 = reinterpret_cast<uint32_t&>(uif.f) ; 
    unsigned u2 = reinterpret_cast<unsigned&>(uif.f) ; 
    LOG(info) << " uif.f " << uif.f << " u2 " << u2 ; 

    bool u_expect = u2 == u  ;
    assert( u_expect ); 
    if(!u_expect) std::raise(SIGINT) ; 
}

void test_reinterpret_cast_arg()
{
    uif_t uif ; 
    uif.u = 42u  ; 
    LOG(info) << " uif.f " << uif.f << " uif.u " << uif.u  ; 
    dummy_optixTrace(  reinterpret_cast<unsigned&>(uif.f) ) ; 
    LOG(info) << " uif.f " << uif.f << " uif.u " << uif.u  ; 
    assert( uif.u == 142u  );  
}
#endif

void test_unsigned_as_double()
{
    unsigned x = 42u ; 
    unsigned y = 420u ; 

    double d = SPack::unsigned_as_double(x, y ); 

    unsigned x2, y2 ; 
    SPack::double_as_unsigned(x2, y2, d ); 
    assert( x2 == x );
    assert( y2 == y );

    LOG(info) << " d " << d << " x2 " << x2 << " y2 " << y2 ; 
}




int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    //test_Encode();  

    //test_Encode_Decode();  
    //test_Encode_Decode_ptr();  
    //test_Encode13_Decode13();  
    //test_Encode22_Decode22();  

    //test_int_as_float(); 
    //test_uint_as_float(); 

    //test_Encode_Decode_unsigned();  
    //test_IsLittleEndian();  

    //test_unsigned_as_int(); 
    //test_unsigned_as_int_16(); 

    //test_Encode22hilo_Decode22hilo(); 

    //test_reinterpret_cast(); 
    //test_reinterpret_cast_arg(); 

    test_unsigned_as_double(); 

    return 0  ; 
}

// om-;TEST=SPackTest om-t

