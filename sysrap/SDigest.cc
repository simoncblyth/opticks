/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "SDigest.hh"
#include "SSys.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <iostream>

#include "PLOG.hh"

const plog::Severity SDigest::LEVEL = PLOG::EnvLevel("SDigest", "DEBUG") ; 


/**


Byte range covering multiple bufsize::

           ---------- 
      i0   ~~~~~~~~~~
           ----------  
           
           ----------  
      i1   ~~~~~~~~~~
           ----------  
      
Byte range within one bufsize::


           ---------- 
      i0   ~~~~~~~~~~
           
      i1   ~~~~~~~~~~
           ----------  
 


           ----------  

**/

std::string SDigest::DigestPathInByteRange(const char* path, int i0, int i1, unsigned bufsize)
{
    LOG(LEVEL) 
        << " path " << path 
        << " i0 " << i0 
        << " i1 " << i1 
        << " bufsize " << bufsize
        ; 

    FILE* fp = fopen(path, "rb");
    if (fp == NULL) 
    {
        LOG(error) << "failed to open path [" << path << "]"  ; 
        return "" ; 
    }
    SDigest dig ; 
    char* data = new char[bufsize] ; 
    int bytes ; 

    int beg = 0 ; // byte index of beginning of buffer in the file
    int end = 0 ; 
    int tot = 0 ;  

    assert( i1 > i0 ) ; 

    while ((bytes = fread (data, 1, bufsize, fp)) != 0) 
    {
        end = beg + bytes ;  

        bool starts = i0 >= beg && i0 < end ;  // capture starts within this bufsize full  
        bool ends   = i1 <= end ;              // capture ends within this bufsize full 

        int idx0(-1) ;  
        int idx1(-1) ;  

        if( starts && ends )
        {
            idx0 = i0 - beg ;   
            idx1 = i1 - beg ;   
        }
        else if( starts && !ends )
        {
            idx0 = i0 - beg ;   
            idx1 = bytes ;   
        }
        else if( !starts && ends )
        { 
            idx0 = 0 ; 
            idx1 = i1 - beg ; 
        }
        else if( !starts && !ends && tot > 0 )  // entire buffer goes to update
        {
            idx0 = 0 ; 
            idx1 = bytes ;   
        } 


        int nup =  idx1 - idx0 ; 
        bool update = idx0 > -1 && idx1 > -1 ; 

        std::string x = update ? SSys::xxd( data+idx0, nup ) : "-" ; 

        LOG(LEVEL)
           << " bytes " << std::setw(8) << bytes 
           << " beg "  << std::setw(8) << beg
           << " end "  << std::setw(8) << end
           << " idx0 " << std::setw(8) << idx0 
           << " idx1 " << std::setw(8) << idx1
           << " nup "  << std::setw(8) << nup
           << ( starts ? " S " : "   ")
           << ( ends ?   " E " : "   ")
           << ( update ? " U " : "   ")
           << " x " << x  
           ; 

        if(update)
        {
            dig.update(data+idx0, nup );   
            tot += nup ;  
        } 

        if( ends ) break ; 

        beg += bytes ;   
    }

    delete[] data ; 

    std::string sdig =  dig.finalize();
    LOG(LEVEL) << " sdig " << sdig ; 
    return sdig ; 
}



std::string SDigest::DigestPath(const char* path, unsigned bufsize)
{
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) 
    {
        std::cerr << "failed to open path [" << path << "]" <<  std::endl ; 
        return "" ; 
    }
    SDigest dig ; 
    char* data = new char[bufsize] ; 
    int bytes ; 
    while ((bytes = fread (data, 1, bufsize, fp)) != 0) dig.update(data, bytes);   
    // NB must update just with the bytes read, not the bufsize
    delete[] data ; 
    return dig.finalize();
}

// https://stackoverflow.com/questions/10324611/how-to-calculate-the-md5-hash-of-a-large-file-in-c


std::string SDigest::DigestPath2(const char* path)
{
    int i;
    FILE* fp = fopen (path, "rb");
    MD5_CTX mdContext;
    int bytes;
    char data[8192];

    if (fp == NULL) {
        printf ("%s can't be opened.\n", path);
        return "";
    }

    MD5_Init (&mdContext);
    while ((bytes = fread (data, 1, 8192, fp)) != 0)
        MD5_Update (&mdContext, data, bytes);

    fclose (fp);
 
    assert( MD5_DIGEST_LENGTH == 16 ); 
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5_Final (digest,&mdContext);
    char *out = (char*)malloc(MD5_DIGEST_LENGTH*2+1);
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) snprintf(&(out[i*2]), MD5_DIGEST_LENGTH*2, "%02x", (unsigned int)digest[i]);
    out[MD5_DIGEST_LENGTH*2] = '\0' ; 
 
    return out ;
}




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


std::string SDigest::Buffer(const char *buffer, int length) 
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


const char* SDigest::hexchar = "0123456789abcdef" ;  

bool SDigest::IsDigest(const char* s)
{
    if( s == NULL ) return false ; 
    if( strlen(s) != 32 ) return false ;  
    for(int i=0 ; i < 32 ; i++ )
    {
        char c = *(s + i) ; 
        if(strchr(hexchar,c) == NULL) return false  ;
    }
    return true  ; 
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

void SDigest::update_str(const char* str )
{
    md5digest_str2md5_update(m_ctx, (char*)str, strlen(str) );
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

const char* SDigest::md5digest_( const char* buffer, int len )
{
    return md5digest_str2md5_monolithic(buffer, len);
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





