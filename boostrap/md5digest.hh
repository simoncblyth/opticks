#ifndef MD5DIGEST_H
#define MD5DIGEST_H

/*
   http://stackoverflow.com/questions/7627723/how-to-create-a-md5-hash-of-a-string-in-c

   hails from env/base/hash/md5digest.h
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#  define COMMON_DIGEST_FOR_OPENSSL
#  include <CommonCrypto/CommonDigest.h>
#  define SHA1 CC_SHA1
#else
#  include <openssl/md5.h>
#endif



extern char* md5digest_str2md5(char* buffer, int length) ;
extern void md5digest_str2md5_update(MD5_CTX& ctx, char* buffer, int length) ;
extern char* md5digest_str2md5_finalize(MD5_CTX& ctx) ;


class MD5Digest 
{
   public:
       MD5Digest();
       virtual ~MD5Digest();
   public:
       void update(char* buffer, int length);
       char* finalize();
   private:
       MD5_CTX m_ctx ;

};








#endif
