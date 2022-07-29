#pragma once
/**
sdigest.h
==========

Header-only hexdigest 

**/

#include <string>

#if defined __APPLE__

#   define COMMON_DIGEST_FOR_OPENSSL
#   include <CommonCrypto/CommonDigest.h>
#   define SHA1 CC_SHA1

#elif defined _MSC_VER

#   include "md5.hh"

#elif __linux

#   include <openssl/md5.h>

#endif

struct NP ; 


struct sdigest
{
    MD5_CTX ctx ; 
    sdigest(); 
    void add( const std::string& str); 
    void add( const char* str ); 
    void add( int i ); 
    void add( const char* buffer, int length); 
    std::string finalize() ; 


    static std::string Item(const NP* a, int i=-1, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1); 
    static std::string Buf(const char* buffer, int length); 
    static std::string Int(int i); 

    static void Update( MD5_CTX& c, const std::string& str); 
    static void Update( MD5_CTX& c, const char* str ); 
    static void Update( MD5_CTX& c, int i ); 
    static void Update( MD5_CTX& c, const char* buffer, int length); 

    static std::string Finalize(MD5_CTX& c); 
}; 


inline sdigest::sdigest(){ MD5_Init(&ctx); }
inline void sdigest::add( const std::string& str){ Update(ctx, str) ; }
inline void sdigest::add( const char* str ){ Update(ctx, str) ; }
inline void sdigest::add( int i ){ Update(ctx, i ) ; }
inline void sdigest::add( const char* str, int length ){ Update(ctx, str, length ) ; }
inline std::string sdigest::finalize(){ return Finalize(ctx) ; } 






#include "NP.hh"
   
inline std::string sdigest::Item( const NP* a, int i, int j, int k, int l, int m, int o ) // static   
{
    const char* start = nullptr ; 
    unsigned num_bytes = 0 ; 
    a->itembytes_(&start, num_bytes, i, j, k, l, m, o ); 
    assert( start && num_bytes > 0 ); 
    return Buf( start, num_bytes ); 
}

inline std::string sdigest::Buf(const char* buffer, int length) // static 
{
    MD5_CTX c;
    MD5_Init(&c);
    Update(c, buffer, length); 
    return Finalize(c); 
}

inline std::string sdigest::Int(int i) // static 
{
    MD5_CTX c;
    MD5_Init(&c);
    Update(c, i); 
    return Finalize(c); 
}

inline void sdigest::Update(MD5_CTX& c, const std::string& str)
{
    Update(c, str.c_str() ); 
}

inline void sdigest::Update(MD5_CTX& c, const char* str )
{
    Update( c, (char*)str, strlen(str) ); 
}

inline void sdigest::Update(MD5_CTX& c, int i )
{
    Update( c, (char*)&i, sizeof(int) ); 
}



inline void sdigest::Update(MD5_CTX& c, const char* buffer, int length) // static
{
    const int blocksize = 512 ; 
    while (length > 0) 
    {
        if (length > blocksize) {
            MD5_Update(&c, buffer, blocksize);
        } else {
            MD5_Update(&c, buffer, length);
        }
        length -= blocksize ;
        buffer += blocksize ;
    }
}

inline std::string sdigest::Finalize(MD5_CTX& c) // static
{
    unsigned char digest[16];
    MD5_Final(digest, &c);

    // 16 binary bytes, into 32 char hex string

    char buf[32+1] ; 
    for (int n = 0; n < 16; ++n) std::snprintf( &buf[2*n], 32+1, "%02x", (unsigned int)digest[n]) ;
    buf[32] = '\0' ; 

    return std::string(buf, buf + 32); 
}


