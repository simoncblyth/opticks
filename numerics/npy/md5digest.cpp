#include "md5digest.hpp"


char* md5digest_str2md5(char* buffer, int length) 
{
    // user should free the returned string digest 

    MD5_CTX ctx;
    MD5_Init(&ctx);

    md5digest_str2md5_update(ctx, buffer, length );

    return md5digest_str2md5_finalize(ctx);
}

void md5digest_str2md5_update(MD5_CTX& ctx, char* buffer, int length) 
{
    const int blocksize = 512 ; 
    while (length > 0) 
    {
        if (length > blocksize) {
            MD5_Update(&ctx, buffer, blocksize);
        } else {
            MD5_Update(&ctx, buffer, length);
        }
        length -= blocksize ;
        buffer += blocksize ;
    }
}

char* md5digest_str2md5_finalize( MD5_CTX& ctx )
{
    unsigned char digest[16];
    MD5_Final(digest, &ctx);

    // 16 binary bytes, into 32 char hex string
    char *out = (char*)malloc(32+1);
    for (int n = 0; n < 16; ++n) snprintf(&(out[n*2]), 16*2, "%02x", (unsigned int)digest[n]);
    out[32] = '\0' ;
    return out;
}




MD5Digest::MD5Digest()
{
   MD5_Init(&m_ctx); 
}
MD5Digest::~MD5Digest()
{
}

void MD5Digest::update(char* buffer, int length)
{
    md5digest_str2md5_update(m_ctx, buffer, length );
}

char* MD5Digest::finalize()
{
    return md5digest_str2md5_finalize(m_ctx);  
}






