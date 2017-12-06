#include "SDigest.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>


char* md5digest_str2md5_monolithic(const char *buffer, int length) 
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

char* md5digest_str2md5(char* buffer, int length) 
{
    // user should free the returned string digest 

    MD5_CTX ctx;
    MD5_Init(&ctx);

    md5digest_str2md5_update(ctx, buffer, length );

    return md5digest_str2md5_finalize(ctx);
}



SDigest::SDigest()
{
   MD5_Init(&m_ctx); 
}
SDigest::~SDigest()
{
}

void SDigest::update(const std::string& str)
{
    update( (char*)str.c_str(), strlen(str.c_str()) );
}

void SDigest::update(char* buffer, int length)
{
    md5digest_str2md5_update(m_ctx, buffer, length );
}

char* SDigest::finalize()
{
    return md5digest_str2md5_finalize(m_ctx);  
}



std::string SDigest::md5digest( const char* buffer, int len )
{
    char* out = md5digest_str2md5_monolithic(buffer, len);
    std::string digest(out);
    free(out);
    return digest;
}



std::string SDigest::digest( void* buffer, int len )
{
    SDigest dig ; 
    dig.update( (char*)buffer, len );
    return dig.finalize();
}

std::string SDigest::digest( std::vector<std::string>& ss)
{
    SDigest dig ; 
    for(unsigned i=0 ; i < ss.size() ; i++) dig.update( ss[i] ) ;
    return dig.finalize();
}


std::string SDigest::digest_skipdupe( std::vector<std::string>& ss)
{
    SDigest dig ; 
    for(unsigned i=0 ; i < ss.size() ; i++) 
    {
        if( i > 0 && ss[i].compare(ss[i-1].c_str()) == 0 ) continue ;    
        dig.update( ss[i] ) ;
    }
    return dig.finalize();
}




