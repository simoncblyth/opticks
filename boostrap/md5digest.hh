#ifndef MD5DIGEST_H
#define MD5DIGEST_H

/*
   http://stackoverflow.com/questions/7627723/how-to-create-a-md5-hash-of-a-string-in-c

   hails from env/base/hash/md5digest.h
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "BRAP_API_EXPORT.h"

#include "md5crossplatform.h"

extern char* md5digest_str2md5(char* buffer, int length) ;
extern void md5digest_str2md5_update(MD5_CTX& ctx, char* buffer, int length) ;
extern char* md5digest_str2md5_finalize(MD5_CTX& ctx) ;


class BRAP_API MD5Digest 
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
