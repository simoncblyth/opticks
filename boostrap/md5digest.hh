#pragma once

/*
   http://stackoverflow.com/questions/7627723/how-to-create-a-md5-hash-of-a-string-in-c

   hails from env/base/hash/md5digest.h
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>


#if defined __APPLE__

#   define BRAP_API 
#   define COMMON_DIGEST_FOR_OPENSSL
#   include <CommonCrypto/CommonDigest.h>
#   define SHA1 CC_SHA1

#elif defined _MSC_VER

#   include "BRAP_API_EXPORT.hh"
#   include "md5.hh"

#elif __linux

#   define BRAP_API 
#   include <openssl/md5.h>

#endif


class BRAP_API MD5Digest 
{
   public:
       static std::string md5digest( const char* buffer, int len );
       template<typename T>
       static std::string arraydigest( T* data, unsigned int n );
   public:
       MD5Digest();
       virtual ~MD5Digest();
   public:
       void update(char* buffer, int length);
       char* finalize();
   private:
       MD5_CTX m_ctx ;

};




