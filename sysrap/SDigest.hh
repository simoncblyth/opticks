#pragma once

/*
   http://stackoverflow.com/questions/7627723/how-to-create-a-md5-hash-of-a-string-in-c
   hails from env/base/hash/md5digest.h
*/

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

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SDigest 
{
   public:
       static std::string md5digest( const char* buffer, int len );
       static std::string digest( void* buffer, int len );
   public:
       SDigest();
       virtual ~SDigest();
   public:
       void update(char* buffer, int length);
       void update( const std::string& str );
       char* finalize();
   private:
       MD5_CTX m_ctx ;

};




