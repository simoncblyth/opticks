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


struct sdigest
{
    static std::string buf(const char* buffer, int length); 
}; 

inline std::string sdigest::buf(const char* buffer, int length)
{
    MD5_CTX c;
    MD5_Init(&c);

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

    unsigned char digest[16];
    MD5_Final(digest, &c);

    // 16 binary bytes, into 32 char hex string

    char buf[32+1] ; 
    for (int n = 0; n < 16; ++n) std::snprintf( &buf[2*n], 32+1, "%02x", (unsigned int)digest[n]) ;
    buf[32] = '\0' ; 

    return std::string(buf, buf + 32); 
}



