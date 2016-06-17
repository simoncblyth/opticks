#pragma once

/*
   http://stackoverflow.com/questions/7627723/how-to-create-a-md5-hash-of-a-string-in-c
   hails from env/base/hash/md5digest.h
*/

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

#include "BRAP_FLAGS.hh"


class BRAP_API BDigest 
{
   public:
       static std::string md5digest( const char* buffer, int len );
   public:
       BDigest();
       virtual ~BDigest();
   public:
       void update(char* buffer, int length);
       char* finalize();
   private:
       MD5_CTX m_ctx ;

};




