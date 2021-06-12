/**
uint128_setbyte
-------------------

uint4 is 4*32bit = 128bit =  16*8 bit = 16 bytes

        u.x       u.y      u.z      u.w   
       +--------+--------+--------+--------+
    s  | f e d c| b a 9 8| 7 6 5 4| 3 2 1 0| 
       +--------+--------+--------+--------+
   bs  |       3|       2|       1|       0|
       +--------+--------+--------+--------+
   us  | 3 2 1 0| 3 2 1 0| 3 2 1 0| 3 2 1 0|    
       +--------+--------+--------+--------+

**/


void uint128_setbyte( uint4& u, signed char c, unsigned s )
{
    unsigned bs = s >> 2 ;       // 0->15 to 0->3 
    unsigned us = s - 4*bs ; 
    unsigned c8 = c & 0xff ;  

    switch(bs)
    {
        case 0: u.x |= c8 << (8*us) ; break ; 
        case 1: u.y |= c8 << (8*us) ; break ; 
        case 2: u.z |= c8 << (8*us) ; break ; 
        case 3: u.w |= c8 << (8*us) ; break ; 
    }
}   

void uint128_getbyte( const uint4& u, signed char& c, unsigned s )
{
    unsigned bs = s >> 2 ;       // 0->15 to 0->3 
    unsigned us = s - 4*bs ; 
    unsigned c8 = 0u ; 

    switch(bs)
    {
        case 0: c8 = (u.x >> (8*us)) & 0xff ; break ; 
        case 1: c8 = (u.y >> (8*us)) & 0xff ; break ; 
        case 2: c8 = (u.z >> (8*us)) & 0xff ; break ; 
        case 3: c8 = (u.w >> (8*us)) & 0xff ; break ; 
    }
    c = c8 ; 
}   



union uint4c16 
{
    uint4 u ; 
    signed char c[16] ; 
};





