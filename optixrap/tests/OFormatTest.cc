#include "OPTICKS_LOG.hh"
#include "OXPPNS.hh"
#include "NPY.hpp"
#include "OFormat.hh"



void test_Get_float()
{
    LOG(info); 
    RTformat f1 = OFormat::Get<float>(1); 
    RTformat f2 = OFormat::Get<float>(2); 
    RTformat f3 = OFormat::Get<float>(3); 
    RTformat f4 = OFormat::Get<float>(4); 

    assert( f1 == RT_FORMAT_FLOAT ); 
    assert( f2 == RT_FORMAT_FLOAT2 ); 
    assert( f3 == RT_FORMAT_FLOAT3 ); 
    assert( f4 == RT_FORMAT_FLOAT4 ); 
}

void test_Get_uchar()
{
    LOG(info); 
    RTformat f1 = OFormat::Get<unsigned char>(1); 
    RTformat f2 = OFormat::Get<unsigned char>(2); 
    RTformat f3 = OFormat::Get<unsigned char>(3); 
    RTformat f4 = OFormat::Get<unsigned char>(4); 

    assert( f1 == RT_FORMAT_UNSIGNED_BYTE ); 
    assert( f2 == RT_FORMAT_UNSIGNED_BYTE2 ); 
    assert( f3 == RT_FORMAT_UNSIGNED_BYTE3 ); 
    assert( f4 == RT_FORMAT_UNSIGNED_BYTE4 ); 
}

void test_ArrayType()
{
    LOG(info); 

    NPY<float>* arr = NPY<float>::make(10,4) ; 

    RTformat f4 = OFormat::ArrayType(arr); 
    assert( f4 == RT_FORMAT_FLOAT4 );   
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_Get_float(); 
    test_Get_uchar(); 
    test_ArrayType();


    return 0 ; 
}


