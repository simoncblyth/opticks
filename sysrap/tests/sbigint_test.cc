/* Copyright (C) Arm Limited, 2023 All rights reserved.

https://developer.arm.com/documentation/ka004805/latest/

The example code is provided to you as an aid to learning when working
with Arm-based technology, including but not limited to programming tutorials.
Arm hereby grants to you, subject to the terms and conditions of this Licence,
a non-exclusive, non-transferable, non-sub-licensable, free-of-charge licence,
to use and copy the Software solely for the purpose of demonstration and
evaluation.

You accept that the Software has not been tested by Arm therefore the Software
is provided "as is", without warranty of any kind, express or implied. In no
event shall the authors or copyright holders be liable for any claim, damages
or other liability, whether in action or contract, tort or otherwise, arising
from, out of or in connection with the Software or the use of Software. */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Macro to help create 128-bit literals from two 64-bit literals
#define MAKE128CONST(hi,lo) ((((__uint128_t)hi << 64) | lo))

// Helper type to use to print 128-bit literals as hexadecimal values
typedef union
{
    __uint128_t as_128;
    struct
    {
        uint64_t lo;
        uint64_t hi;
    }  as_2x64;
} reinterpret128_t;

// Function to write a 128-bit value to memory
__attribute__((always_inline)) static void reg128_write(uint64_t addr, __uint128_t data)
{
    volatile __uint128_t *as_ptr = (volatile __uint128_t *)addr;
    *as_ptr = data;
}

// Function to read a 128-bit value from memory
__attribute((always_inline)) static __uint128_t reg128_read(uint64_t addr)
{
    volatile __uint128_t *as_ptr = (volatile __uint128_t *)addr;
    return *as_ptr;
}

// Function to convert a 128-bit value to a zero-padded hexadecimcal string
__attribute__((always_inline)) static const char * reg128_as_hex(__uint128_t data)
{
    reinterpret128_t temp = *((reinterpret128_t *)(&data));
    char * buffer = (char*)malloc(33);
    sprintf(buffer, "%016llx%016llx", temp.as_2x64.hi, temp.as_2x64.lo);
    return buffer;
}

int main(void)
{
    // 128-bit literals are not supported. Instead, use the MAKE128CONST macro.
    // __uint128_t data = 0x44444444333333332222222211111111;

    uint64_t buffer[2] ; 
    uint64_t* ptr = &buffer[0] ; 
    uint64_t iptr = (uint64_t)(ptr) ; 

    // Write 128-bit constant value to an address.
    reg128_write(iptr, MAKE128CONST(0x4444444433333333ULL, 0x2222222211111111ULL));

    // Print 128-bit value read from an address.
    // The helper function reg128_as_hex is used to convert a 128-bit value to a hex string without the '0x' suffix.
    printf("0x%s\n", reg128_as_hex(reg128_read(iptr)));

    return 0;
}

