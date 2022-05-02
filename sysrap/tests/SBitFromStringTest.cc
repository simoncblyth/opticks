#include "OPTICKS_LOG.hh"
#include "SBit.hh"

void test_FromString_0()
{
    unsigned long long ull_0 = SBit::FromString("0b11111111") ; 
    LOG(info) << " ull_0 " << ull_0 ; 
    assert( ull_0 == 255 ); 
    unsigned long long ull_1 = SBit::FromString("0xff") ; 
    assert( ull_1 == 255 ); 
    unsigned long long ull_2 = SBit::FromString("255") ; 
    assert( ull_2 == 255 ); 
    unsigned long long ull_3 = SBit::FromString("0,1,2,3,4,5,6,7") ; 
    LOG(info) << " ull_3 " << ull_3 ; 
    assert( ull_3 == 255 ); 
}



const char* EXAMPLES  = R"LITERAL(

0p
,
0
0d0
0x0
0x000000000
0b0
0b00000000

0,
0p0
1
0x1
0b1
0b000000001

1,
0p1
2
0x2
0b10
0b0010

0,1
0p0,1
3
0d3
0x3
0b11
0b0000011

2,
0p2
4
0d4
0x4
0b100
0b0000100


0,2
0p0,2
5
0d5
0x5
0b101
0b00000101


# Complemented zero : ie all 64 bits set  

~0


# PosString setting only the comma delimited bitpos ranging from 0 to 15

0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,

# PosString all bits set other than the comma delimited bitpos ranging from 0 to 15

~0,
~1,
~2,
~3,
~4,
~5,
~6,
~7,
~8,
~9,
~10,
~11,
~12,
~13,
~14,
~15,


# alt tilde avoiding shell escaping

t0,
t1,

2
0d2
0x2
~0b1101
0b0010
1,
0p1,

t0b1101

t1,
t0p1

~8,8


)LITERAL";

void test_FromString_(const char* str)
{
    if(strlen(str) == 0)
    {
        std::cout << std::endl ; 
    }
    else if(str[0] == '#')
    {
        std::cout << str << std::endl ; 
    }
    else
    {
        unsigned long long ull = SBit::FromString(str) ; 
        std::cout
            << std::setw(15) << str 
            << " : "
            << " "      << std::setw(20) << SBit::String(ull)
            << " (0p) " << std::setw(10) << SBit::PosString(ull)
            << " (0x) " << std::setw(16) << SBit::HexString(ull) 
            << " (0d) " << std::setw(20) << SBit::DecString(ull)
            << " (0b) " << std::setw(20) << SBit::BinString(ull)
            << std::endl  
            ;
    }
}


void test_FromString()
{
    LOG(info); 
    std::stringstream ss ; 
    ss.str(EXAMPLES); 
    std::string s;
    while (std::getline(ss, s, '\n')) 
    {
        const char* str = s.c_str(); 
        test_FromString_(str); 
    }
}

void test_FromString_args(int argc, char** argv)
{
    LOG(info) << " from arguments argc " << argc ;  
    for(int i=1 ; i < argc ; i++) std::cout << std::setw(5) << i << " : " << argv[i] << std::endl  ; 
    for(int i=1 ; i < argc ; i++) test_FromString_(argv[i]) ; 
}

void test_FromEString()
{
    unsigned long long emm = SBit::FromEString("EMM"); 
    LOG(info) << " emm " << SBit::HexString(emm) << " 0x" << std::hex << emm << std::dec << std::endl ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    /*
    test_FromString();  
    test_FromString_args(argc, argv);  
    */ 

    test_FromEString();  


    return 0 ; 
}

