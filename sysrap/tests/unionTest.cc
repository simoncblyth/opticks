// name=unionTest ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

/**

https://stackoverflow.com/questions/252552/why-do-we-need-c-unions

**/


typedef union
{
    struct {
        unsigned char byte0 ;
        unsigned char byte1 ;
        unsigned char byte2 ;
        unsigned char byte3 ;
    } bytes;
    unsigned int dword;
} HW_Register;


union HW_Register2
{
    struct {
        unsigned char byte0 ;
        unsigned char byte1 ;
        unsigned char byte2 ;
        unsigned char byte3 ;
    } bytes ;
    unsigned int dword;
}; 


#include <cstdio>

int main()
{
    //HW_Register reg;
    HW_Register2 reg;

    reg.dword = 0x12345678;

    printf("// reg.dword        %x \n" , reg.dword ); 
    printf("// reg.bytes.byte0  %x \n" , reg.bytes.byte0 ); 
    printf("// reg.bytes.byte1  %x \n" , reg.bytes.byte1 ); 
    printf("// reg.bytes.byte2  %x \n" , reg.bytes.byte2 ); 
    printf("// reg.bytes.byte3  %x \n" , reg.bytes.byte3 ); 

    return 0 ; 
}

