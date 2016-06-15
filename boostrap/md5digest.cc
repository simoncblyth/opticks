#include "md5digest.hh"




char* md5digest_str2md5_monlithic(const char *buffer, int length) 
{
    // user should free the returned string digest 

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
    char *out = (char*)malloc(32+1);
    for (int n = 0; n < 16; ++n) snprintf(&(out[n*2]), 16*2, "%02x", (unsigned int)digest[n]);
    out[32] = '\0' ;
    return out;
}


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



std::string MD5Digest::md5digest( const char* buffer, int len )
{
    char* out = md5digest_str2md5_monolithic(buffer, len);
    std::string digest(out);
    free(out);
    return digest;
}


template<typename T>
std::string MD5Digest::arraydigest( T* data, unsigned int n )
{
    return md5digest( (char*)data, sizeof(T)*n );
}





template std::string MD5Digest::arraydigest( float*, unsigned int);
template std::string MD5Digest::arraydigest( int*, unsigned int);
template std::string MD5Digest::arraydigest( unsigned int*, unsigned int);






