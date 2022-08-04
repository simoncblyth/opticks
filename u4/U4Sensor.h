#pragma once
/**
U4Sensor.h
============

Pure virtual protocol base used to interface detector
specific characteristics of sensors with Opticks. 


getId
    method is called on the outer volume of every factorized instance during geometry translation, 
    the returned unsigned value is used by IAS_Builder to set the OptixInstance .instanceId 
    Within CSGOptiX/CSGOptiX7.cu:: __closesthit__ch *optixGetInstanceId()* is used to 
    passes the instanceId value into "quad2* prd" (per-ray-data) which is available 
    within qudarap/qsim.h methods. 
    
    The 32 bit unsigned returned by *getInstanceIdentity* may not use the top 8 bits 
    because of an OptiX 7 limit of 24 bits, from Properties::dump::

        limitMaxInstanceId :   16777215    ffffff

    (that limit might well be raised in versions after 700)


HMM: how to split those 24 bits ? 

1. sensor id
2. sensor category (4 cat:2 bits, 8 cat: 3 bits)

::

    In [14]: for i in range(32): print(" (0x1 << %2d) - 1   %16x   %16d  %16.2f  " % (i, (0x1 << i)-1, (0x1 << i)-1, float((0x1 << i)-1)/1e6 )) 

     (0x1 <<  0) - 1                  0                  0              0.00  
     (0x1 <<  1) - 1                  1                  1              0.00  
     (0x1 <<  2) - 1                  3                  3              0.00  
     (0x1 <<  3) - 1                  7                  7              0.00  
     (0x1 <<  4) - 1                  f                 15              0.00  
     (0x1 <<  5) - 1                 1f                 31              0.00  
     (0x1 <<  6) - 1                 3f                 63              0.00  
     (0x1 <<  7) - 1                 7f                127              0.00  
     (0x1 <<  8) - 1                 ff                255              0.00  
     (0x1 <<  9) - 1                1ff                511              0.00  
     (0x1 << 10) - 1                3ff               1023              0.00  
     (0x1 << 11) - 1                7ff               2047              0.00  
     (0x1 << 12) - 1                fff               4095              0.00  
     (0x1 << 13) - 1               1fff               8191              0.01  
     (0x1 << 14) - 1               3fff              16383              0.02  
     (0x1 << 15) - 1               7fff              32767              0.03  
     (0x1 << 16) - 1               ffff              65535              0.07  
     (0x1 << 17) - 1              1ffff             131071              0.13  
     (0x1 << 18) - 1              3ffff             262143              0.26  
     (0x1 << 19) - 1              7ffff             524287              0.52  
     (0x1 << 20) - 1              fffff            1048575              1.05  
     (0x1 << 21) - 1             1fffff            2097151              2.10  
     (0x1 << 22) - 1             3fffff            4194303              4.19  
     (0x1 << 23) - 1             7fffff            8388607              8.39  
     (0x1 << 24) - 1             ffffff           16777215             16.78  
     (0x1 << 25) - 1            1ffffff           33554431             33.55  
     (0x1 << 26) - 1            3ffffff           67108863             67.11  
     (0x1 << 27) - 1            7ffffff          134217727            134.22  
     (0x1 << 28) - 1            fffffff          268435455            268.44  
     (0x1 << 29) - 1           1fffffff          536870911            536.87  
     (0x1 << 30) - 1           3fffffff         1073741823           1073.74  
     (0x1 << 31) - 1           7fffffff         2147483647           2147.48  





**/
class G4PVPlacement ; 

struct U4Sensor
{
    virtual unsigned getId(             const G4PVPlacement* pv) const = 0 ; 
    virtual float    getEfficiency(     const G4PVPlacement* pv) const = 0 ; 
    virtual float    getEfficiencyScale(const G4PVPlacement* pv) const = 0 ; 
}; 


