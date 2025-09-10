#pragma once
/**
NP_CURL.h : Remote array transformation over HTTP using libcurl
==================================================================

libcurl based upload/download of NP.hh arrays to a remote API service such
as the FastAPI implemented endpoint started by::

   cd /usr/local/env/fastapi_check
   ./dev.sh   # symbolic link to ~/env/tools/fastapi_check/dev.sh

Usage::

    NP* b = NP_CURL::TransformRemote(a);

See::

    ~/np/tests/np_curl_test/np_curl_test.sh

TODO
----

1. avoid duplication of memory for the download
2. stress test with large arrays to find memory leaks

**/

#include <iomanip>
#include <sstream>

#include <cassert>
#include <cstring>

#include <curl/curl.h>

#include "NP_CURL_Header.h"
#include "NP_CURL_Upload.h"
#include "NP_CURL_Download.h"

struct U ;
struct NP ;


struct NP_CURL
{
    static NP_CURL* INSTANCE ;

    static constexpr const char* NP_CURL_API_URL = "NP_CURL_API_URL" ;
    static constexpr const char* NP_CURL_API_URL_DEFAULT = "http://127.0.0.1:8000/array_transform" ;

    static constexpr const char* NP_CURL_API_TOKEN = "NP_CURL_API_TOKEN" ;
    static constexpr const char* NP_CURL_API_TOKEN_DEFAULT = "secret" ;

    static constexpr const char* NP_CURL_API_LEVEL = "NP_CURL_API_LEVEL" ;
    static constexpr int         NP_CURL_API_LEVEL_DEFAULT = 0 ;


    const char*       url ;
    const char*       token ;
    int               level ;

    CURL*              session;
    NP_CURL_Upload*    upload ;
    NP_CURL_Download*  download ;
    NP_CURL_Header     uhdr ;
    NP_CURL_Header     dhdr ;

    CURLcode           curl_code ;
    long               http_code ;
    int                rc ;


    static  NP_CURL* Get();
    static  NP* TransformRemote( const NP* a );
    static void Clear();

    NP_CURL();
    virtual ~NP_CURL();

    NP* transformRemote( const NP* a );

    void prepare_upload( const NP* a );
    void prepare_download();
    void perform();
    NP*  collect_download();


    std::string desc() const ;
};

inline NP_CURL* NP_CURL::INSTANCE = nullptr ;

inline NP_CURL* NP_CURL::Get()
{
    return INSTANCE ? INSTANCE : new NP_CURL ;
}


inline NP* NP_CURL::TransformRemote( const NP* a ) // static
{
    NP_CURL* nc = Get();
    return nc->transformRemote( a )  ;
}

inline void NP_CURL::Clear()
{
    NP_CURL* nc = Get();
    delete nc ;
}


inline NP_CURL::NP_CURL()
    :
    url(  U::GetEnv(NP_CURL_API_URL,   NP_CURL_API_URL_DEFAULT )),
    token(U::GetEnv(NP_CURL_API_TOKEN, NP_CURL_API_TOKEN_DEFAULT )),
    level(U::GetEnvInt(NP_CURL_API_LEVEL, NP_CURL_API_LEVEL_DEFAULT)),
    session(nullptr),
    upload(new NP_CURL_Upload),
    download(new NP_CURL_Download),
    uhdr("uhdr"),
    dhdr("dhdr"),
    curl_code((CURLcode)0),
    http_code(0),
    rc(0)
{
    if(level > 0) std::cout << "[NP_CURL::NP_CURL\n" ;

    INSTANCE = this ;
    curl_global_init(CURL_GLOBAL_ALL);
    session = curl_easy_init();

    if(level > 0) std::cout << "]NP_CURL::NP_CURL\n" ;
}

inline NP_CURL::~NP_CURL()
{
    if(level > 0) std::cout << "[NP_CURL::~NP_CURL\n" ;

    curl_easy_cleanup(session);
    curl_global_cleanup();

    if(level > 0) std::cout << "]NP_CURL::~NP_CURL\n" ;
}


inline NP* NP_CURL::transformRemote( const NP* a )
{
    prepare_upload( a );
    prepare_download();
    perform();

    NP* b = collect_download();

    dhdr.clear();
    uhdr.clear();

    return b ;
}

inline void NP_CURL::prepare_upload( const NP* a )
{
    if(level > 0) std::cout << "[NP_CURL::prepare_upload\n" ;

    upload->size = a->arr_bytes();
    upload->data = a->bytes();
    std::string dtype = a->dtype_name() ; // eg float32 uint8 uint16 ..
    std::string shape = a->sstr();        // eg "(10, 4, 4, )"

    uhdr.prepare_upload( dtype.c_str(), shape.c_str(), token, level );

    if(level > 0) std::cout << "-NP_CURL::prepare_upload shape[" << shape << "]\n" ;

    curl_easy_setopt(session, CURLOPT_URL, url );
    curl_easy_setopt(session, CURLOPT_POST, 1L);
    curl_easy_setopt(session, CURLOPT_HTTPHEADER, uhdr.headerlist);

    curl_easy_setopt(session, CURLOPT_READFUNCTION, NP_CURL_Upload::read_callback);
    curl_easy_setopt(session, CURLOPT_READDATA, upload );
    curl_easy_setopt(session, CURLOPT_POSTFIELDSIZE, (long)upload->size);

    if(level > 0) std::cout << "]NP_CURL::prepare_upload\n" ;
}


inline void NP_CURL::prepare_download()
{
    if(level > 0) std::cout << "[NP_CURL::prepare_download\n" ;

    download->buffer = (char*)malloc(1); // Start with a 1-byte buffer
    download->size = 0 ;
    download->buffer[0] = '\0';

    curl_easy_setopt(session, CURLOPT_WRITEFUNCTION, NP_CURL_Download::write_callback);
    curl_easy_setopt(session, CURLOPT_WRITEDATA, download );

    if(level > 0) std::cout << "]NP_CURL::prepare_download\n" ;
}


inline void NP_CURL::perform()
{
    if(level > 0) std::cout << "[NP_CURL::perform\n" ;
    curl_code = curl_easy_perform(session);
    if (curl_code != CURLE_OK)
    {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(curl_code));
        rc = 1 ;
    }

    curl_easy_getinfo(session, CURLINFO_RESPONSE_CODE, &http_code);
    if(level > 0) std::cout << "-NP_CURL::perform http_code[" << http_code << "]\n" ;

    if( http_code >= 400 ) std::cout
        << "-NP_CURL::perform download.buffer[\n"
        << download->buffer
        << "\n]\n"
        ;

    struct curl_header* h;
    struct curl_header* p = nullptr ;
    do
    {
        h = curl_easy_nextheader(session, CURLH_HEADER, -1, p );
        if(h) dhdr.collect(h->name, h->value);
        p = h;
    }
    while(h);

    dhdr.collect_json_content(download->buffer, download->size );

    if(level > 0) std::cout << "]NP_CURL::perform\n" ;
}

/**
NP_CURL::collect_download
---------------------------

Currently memory for the array and the download buffer are duplicated.
If could read the headers to give shape and dtype before the content
could use the array bytes to presize the download
before doing the download ?  That would avoid many realloc when
dealing with large arrays.

**/


inline NP* NP_CURL::collect_download()
{
    if(level > 0) std::cout << "[NP_CURL::collect_download\n" ;
    if( http_code != 200 )
    {
        fprintf(stderr, "collect_download abort:  http_code %ld\n", http_code);
        rc = 2 ;
        return nullptr ;
    }


    std::string dtype = dtype_convert::from_name(dhdr.dtype.c_str()) ;
    if(level > 0) std::cout << "-NP_CURL::collect_download dhdr.dtype[" << dhdr.dtype << "]\n" ;
    if(level > 0) std::cout << "-NP_CURL::collect_download      dtype[" << dtype << "]\n" ;

    NP* b = new NP( dtype.c_str(), dhdr.sh );
    bool expect = b->arr_bytes() == download->size ;

    if( !expect ) std::cerr
        << "-NP_CURL::collect_download "
        << " dhdr.sstr " << dhdr.sstr()
        << " b.arr_bytes " << b->arr_bytes()
        << " download.size " << download->size
        << " expect " << ( expect ? "YES" : "NO " )
        << "\n"
        ;


    assert( expect );
    b->read_bytes( download->buffer );

    if(level > 1) std::cout << desc();
    if(level > 0) std::cout << "]NP_CURL::collect_download\n" ;

    return b ;
}


inline std::string NP_CURL::desc() const
{
    std::stringstream ss ;
    ss
       << "[NP_CURL::desc\n"
       << download->desc() << "\n"
       << dhdr.desc() << "\n"
       << "]NP_CURL::desc\n"
       ;

    std::string str = ss.str() ;
    return str ;
}


