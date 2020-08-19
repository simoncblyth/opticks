#include "OPTICKS_LOG.hh"
#include "OXPPNS.hh"
#include "OFormat.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    RTformat f1 = OFormat::TextureFormat<float>(1); 
    RTformat f2 = OFormat::TextureFormat<float>(2); 
    RTformat f3 = OFormat::TextureFormat<float>(3); 
    RTformat f4 = OFormat::TextureFormat<float>(4); 

    assert( f1 == RT_FORMAT_FLOAT ); 
    assert( f2 == RT_FORMAT_FLOAT2 ); 
    assert( f3 == RT_FORMAT_FLOAT3 ); 
    assert( f4 == RT_FORMAT_FLOAT4 ); 

    RTformat u1 = OFormat::TextureFormat<unsigned char>(1); 
    RTformat u2 = OFormat::TextureFormat<unsigned char>(2); 
    RTformat u3 = OFormat::TextureFormat<unsigned char>(3); 
    RTformat u4 = OFormat::TextureFormat<unsigned char>(4); 

    assert( u1 == RT_FORMAT_UNSIGNED_BYTE ); 
    assert( u2 == RT_FORMAT_UNSIGNED_BYTE2 ); 
    assert( u3 == RT_FORMAT_UNSIGNED_BYTE3 ); 
    assert( u4 == RT_FORMAT_UNSIGNED_BYTE4 ); 

    LOG(info); 

    return 0 ; 
}


