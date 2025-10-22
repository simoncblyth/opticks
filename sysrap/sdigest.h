#pragma once
/**
sdigest.h
==========

Header-only hexdigest

Example from /usr/include/openssl/opensslv.h::

   33 # define OPENSSL_VERSION_NUMBER  0x100020bfL

**/

#include <string>
#include <vector>
#include <array>
#include <sstream>

#if defined __APPLE__

#   define COMMON_DIGEST_FOR_OPENSSL
#   include <CommonCrypto/CommonDigest.h>
#   define SHA1 CC_SHA1

#elif defined _MSC_VER

#   include "md5.hh"

#elif __linux

#   include <openssl/md5.h>
#   include <openssl/opensslv.h>

#endif

struct NP ;


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

struct sdigest
{
    static std::string Desc();

    MD5_CTX ctx ;
    sdigest();
    void add( const std::string& str);
    void add( const char* str );
    void add( int i );
    void add( const char* buffer, int length);
    void add( const std::vector<unsigned char>& bytes );

    std::string finalize() ;
    std::array<unsigned char,16> finalize_raw() ;


    static std::string Item(const NP* a, int i=-1, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1);
    static std::array<unsigned char,16> ItemRaw(const NP* a, int i=-1, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1);

    static std::string Buf(const char* buffer, int length);
    static std::array<unsigned char,16> BufRaw(const char* buffer, int length);
    static void BufRaw_(unsigned char* digest_16, const char* buffer, int length);


    static std::string Int(int i);
    static std::string Path(const char* path, unsigned bufsize=8192 );
    static std::string Paths(std::vector<std::string>& paths, unsigned bufsize=8192 );


    static void Update( MD5_CTX& c, const std::string& str);
    static void Update( MD5_CTX& c, const char* str );
    static void Update( MD5_CTX& c, int i );
    static void Update( MD5_CTX& c, const char* buffer, int length);

    //template<typename T> static void Update_(MD5_CTX& c, T v );
    template<typename T> static void Update_(MD5_CTX& c, T* vv, size_t count );

    //template<typename T> void add_( T v );
    template<typename T> void add_( T* vv, size_t count );

    static std::string DescRaw( unsigned char* digest16 );

    static std::string Finalize(MD5_CTX& c);
    static std::array<unsigned char,16> FinalizeRaw(MD5_CTX& c);

    static void FinalizeRaw_(unsigned char* digest_16, MD5_CTX& c );



};


inline std::string sdigest::Desc()
{
    std::stringstream ss ;
#if OPENSSL_VERSION_NUMBER == 0x100020bfL
    ss << "OPENSSL_VERSION_NUMBER == 0x100020bfL" << std::endl ;
#elif OPENSSL_VERSION_NUMBER > 0x100020bfL
    ss << "OPENSSL_VERSION_NUMBER > 0x100020bfL" << std::endl ;
#elif OPENSSL_VERSION_NUMBER < 0x100020bfL
    ss << "OPENSSL_VERSION_NUMBER < 0x100020bfL" << std::endl ;
#endif

#if __linux
    ss << "Linux : OPENSSL_VERSION_NUMBER is 0x" << std::hex << OPENSSL_VERSION_NUMBER  << std::dec << std::endl ;
#endif

    std::string s = ss.str();
    return s ;
}


inline sdigest::sdigest(){ MD5_Init(&ctx); }
inline void sdigest::add( const std::string& str){ Update(ctx, str) ; }
inline void sdigest::add( const char* str ){ Update(ctx, str) ; }
inline void sdigest::add( int i ){ Update(ctx, i ) ; }
inline void sdigest::add( const char* str, int length ){ Update(ctx, str, length ) ; }
inline void sdigest::add( const std::vector<unsigned char>& bytes ){ Update(ctx, (char*)bytes.data(), bytes.size() ); }

inline std::string sdigest::finalize(){ return Finalize(ctx) ; }
inline std::array<unsigned char,16> sdigest::finalize_raw(){ return FinalizeRaw(ctx) ; }



#include "NP.hh"

inline std::string sdigest::Item( const NP* a, int i, int j, int k, int l, int m, int o ) // static
{
    const char* start = nullptr ;
    NP::INT num_bytes = 0 ;
    a->itembytes_(&start, num_bytes, i, j, k, l, m, o );
    assert( start && num_bytes > 0 );
    return Buf( start, num_bytes );
}

/**
sdigest::ItemRaw
-----------------

For example of using the digests to check for duplicates see 
sysrap/tests/sdigest_duplicate_test/sdigest_duplicate_test.cc

**/


inline std::array<unsigned char,16> sdigest::ItemRaw( const NP* a, int i, int j, int k, int l, int m, int o ) // static
{
    const char* start = nullptr ;
    NP::INT num_bytes = 0 ;
    a->itembytes_(&start, num_bytes, i, j, k, l, m, o );
    assert( start && num_bytes > 0 );
    return BufRaw( start, num_bytes );
}


inline std::string sdigest::Buf(const char* buffer, int length) // static
{
    MD5_CTX c;
    MD5_Init(&c);
    Update(c, buffer, length);
    return Finalize(c);
}
inline std::array<unsigned char,16> sdigest::BufRaw(const char* buffer, int length) // static
{
    MD5_CTX c;
    MD5_Init(&c);
    Update(c, buffer, length);
    return FinalizeRaw(c);
}

inline void sdigest::BufRaw_(unsigned char* digest_16, const char* buffer, int length) // static
{
    MD5_CTX c;
    MD5_Init(&c);
    Update(c, buffer, length);
    FinalizeRaw_(digest_16, c);
}




inline std::string sdigest::Int(int i) // static
{
    MD5_CTX c;
    MD5_Init(&c);
    Update(c, i);
    return Finalize(c);
}


inline std::string sdigest::Path(const char* path, unsigned bufsize )
{
    //std::cout << "sdigest::Path [" << path << "]" << std::endl ;

    FILE* fp = fopen(path, "rb");
    if (fp == NULL)
    {
        std::cerr << "failed to open path [" << path << "]" <<  std::endl ;
        return "" ;
    }

    sdigest dig ;
    char* data = new char[bufsize] ;
    int bytes ;
    while ((bytes = fread (data, 1, bufsize, fp)) != 0) dig.add(data, bytes);
    // NB must update just with the bytes read, not the bufsize
    delete[] data ;

    std::string out = dig.finalize();
    //std::cout << "sdigest::Path out [" << out << "]" << std::endl ;
    return out ;
}

inline std::string sdigest::Paths(std::vector<std::string>& paths, unsigned bufsize )
{
    sdigest dig ;
    char* data = new char[bufsize] ;

    int num_paths = paths.size();
    for(int i=0 ; i < num_paths ; i++)
    {
        const char* path = paths[i].c_str();

        FILE* fp = fopen(path, "rb");
        if (fp == NULL)
        {
            std::cerr
                << "sdigest::Paths"
                << " failed to open"
                << " path [" << path << "]\n"
                ;
            continue ;
        }

        int bytes ;
        while ((bytes = fread (data, 1, bufsize, fp)) != 0) dig.add(data, bytes);

        fclose(fp);
    }

    delete[] data ;

    std::string out = dig.finalize();
    //std::cout << "sdigest::Path out [" << out << "]" << std::endl ;
    return out ;
}





inline void sdigest::Update(MD5_CTX& c, const std::string& str)
{
    Update(c, str.c_str() );
}

inline void sdigest::Update(MD5_CTX& c, const char* str )
{
    Update( c, (char*)str, strlen(str) );
}

inline void sdigest::Update(MD5_CTX& c, int i )
{
    Update( c, (char*)&i, sizeof(int) );
}

inline void sdigest::Update(MD5_CTX& c, const char* buffer, int length) // static
{
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
}



/*
template<typename T>
inline void sdigest::Update_(MD5_CTX& c, T v )
{
    Update( c, (char*)&v, sizeof(T) );
}

template<typename T>
inline void sdigest::add_( T v ){ Update_<T>(ctx, v );  }
*/

template<typename T>
inline void sdigest::Update_(MD5_CTX& c, T* vv, size_t count )
{
    Update( c, (char*)vv, sizeof(T)*count );
}

template<typename T>
inline void sdigest::add_( T* vv, size_t count ){ Update_<T>(ctx, vv, count  );  }



inline std::string sdigest::DescRaw( unsigned char* digest16 )
{
    char buf[32+1] ;
    for (int n = 0; n < 16; ++n) std::snprintf( &buf[2*n], 32+1, "%02x", (unsigned int)digest16[n]) ;
    buf[32] = '\0' ;
    return std::string(buf, buf + 32);
}


inline std::string sdigest::Finalize(MD5_CTX& c) // static
{
    unsigned char digest[16];
    MD5_Final(digest, &c);

    // 16 binary bytes, into 32 char hex string

    char buf[32+1] ;
    for (int n = 0; n < 16; ++n) std::snprintf( &buf[2*n], 32+1, "%02x", (unsigned int)digest[n]) ;
    buf[32] = '\0' ;

    return std::string(buf, buf + 32);
}

inline std::array<unsigned char,16> sdigest::FinalizeRaw(MD5_CTX& c) // static
{
    unsigned char digest[16];
    MD5_Final(digest, &c);

    std::array<unsigned char, 16> result;
    std::copy(digest, digest + 16, result.begin());
    return result;
}

inline void sdigest::FinalizeRaw_(unsigned char* digest_16, MD5_CTX& c ) // static
{
    MD5_Final(digest_16, &c );
}




#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

