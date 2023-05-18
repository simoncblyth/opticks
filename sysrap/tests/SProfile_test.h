#pragma once

template<int> struct SProfile ; 

struct SProfile_test
{
    int d ; 
    SProfile<4>* prof ; 

    SProfile_test(); 
}; 
