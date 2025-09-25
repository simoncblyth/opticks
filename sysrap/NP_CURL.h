#pragma once
/**
NP_CURL.h : Remote array transformation over HTTP using libcurl
==================================================================

libcurl based upload/download of NP.hh arrays to a remote HTTP API service.


Start FastAPI endpoint::

   /usr/local/env/fastapi_check/dev.sh

Make requests::

    ~/np/tests/np_curl_test/np_curl_test.sh
    LEVEL=1 ~/np/tests/np_curl_test/np_curl_test.sh    ## more verbosity

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

#ifdef WITHOUT_MAGIC
#include "NP_CURL_Upload.h"
#include "NP_CURL_Download.h"
#endif

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
    curl_mime*         mime ;

#ifdef WITHOUT_MAGIC
    NP_CURL_Upload_1*  upload ;
    NP_CURL_Download*  download ;
#else
    NP*                download ;
#endif

    NP_CURL_Header     uhdr ;
    NP_CURL_Header     dhdr ;

    CURLcode           curl_code ;
    long               http_code ;
    int                rc ;


    static  NP_CURL* Get();
    static  NP* TransformRemote( NP* a, int index );
    static void Clear();

    NP_CURL();
    virtual ~NP_CURL();

    NP* transformRemote( NP* a, int index );

    void prepare_upload(   NP* a, int index );
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


inline NP* NP_CURL::TransformRemote( NP* a, int index ) // static
{
    if(!a) return nullptr ;
    NP_CURL* nc = Get();
    return nc->transformRemote( a, index )  ;
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
    mime(nullptr),
#ifdef WITHOUT_MAGIC
    upload(new NP_CURL_Upload_1),
    download(new NP_CURL_Download),
#else
    download(nullptr),
#endif
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


inline NP* NP_CURL::transformRemote( NP* a, int index )
{
    if(!a) return nullptr ;

    prepare_upload( a, index );
    prepare_download();
    perform();

    NP* b = collect_download();

    dhdr.clear();
    uhdr.clear();

    return b ;
}

/**
NP_CURL::prepare_upload
-----------------------

* https://curl.se/libcurl/c/CURLOPT_MIMEPOST.html
* https://stackoverflow.com/questions/4007969/application-x-www-form-urlencoded-or-multipart-form-data

There are two types of HTTP POST content::

1. "application/x-www-form-urlencoded" which is only suitable for small arrays
   as the data is url escaped as effectively a giant query string
2. "multipart/form-data" more flexible and efficient for large arrays


TODO:

1. compare performance for large uploads with different options, eg::

   curl_easy_setopt(session, CURLOPT_INFILESIZE_LARGE, (curl_off_t)upload_size );

2. implement seek and cleanup callbacks and test for memory leaks

**/

inline void NP_CURL::prepare_upload( NP* a, int index )
{
    if(level > 0) std::cout << "[NP_CURL::prepare_upload\n" ;

    long upload_size = 0 ;

#ifdef WITHOUT_MAGIC
    upload->size = a->arr_bytes();
    upload->buffer = (char*)a->bytes();
    upload->position = 0 ;
    upload_size = upload->size ;     // just arr data in old approach
    std::string dtype = a->dtype_name() ; // eg float32 uint8 uint16 ..
    std::string shape = a->sstr();        // eg "(10, 4, 4, )"
    uhdr.prepare_upload( token, index, level, dtype.c_str(), shape.c_str() );
#else
    a->update_headers();

    upload_size = a->serialize_bytes() ;  // both hdr and arr data
    uhdr.prepare_upload( token, index, level );
#endif

    if(level > 0) std::cout << "-NP_CURL::prepare_upload\n" ;

    curl_easy_setopt(session, CURLOPT_URL, url );
    curl_easy_setopt(session, CURLOPT_HTTPHEADER, uhdr.headerlist);

    mime = curl_mime_init(session);
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "upload");
    curl_mime_filename(part, "array_data" );

#ifdef WITHOUT_MAGIC
    curl_mime_data_cb(part, upload_size, NP_CURL_Upload_1::read_callback, nullptr, nullptr, upload );
#else
    curl_mime_data_cb(part, upload_size, NP::ReadToBufferCallback, nullptr, nullptr, (void*)a );
#endif

    curl_easy_setopt(session, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(session, CURLOPT_POSTFIELDSIZE, upload_size);

    if(level > 0) std::cout << "]NP_CURL::prepare_upload\n" ;
}



inline void NP_CURL::prepare_download()
{
    if(level > 0) std::cout << "[NP_CURL::prepare_download\n" ;

#ifdef WITHOUT_MAGIC
    download->buffer = (char*)malloc(1); // Start with a 1-byte buffer
    download->size = 0 ;
    download->buffer[0] = '\0';

    curl_easy_setopt(session, CURLOPT_WRITEFUNCTION, NP_CURL_Download::write_callback);
    curl_easy_setopt(session, CURLOPT_WRITEDATA, download );
#else

    download = new NP ;
    download->prepareForStreamIn();

    curl_easy_setopt(session, CURLOPT_WRITEFUNCTION, NP::WriteToArrayCallback );
    curl_easy_setopt(session, CURLOPT_WRITEDATA, download );
#endif

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

    if(mime)
    {
        curl_mime_free(mime);
        mime = nullptr ;
    }


    curl_easy_getinfo(session, CURLINFO_RESPONSE_CODE, &http_code);
    if(level > 0) std::cout << "-NP_CURL::perform http_code[" << http_code << "]\n" ;

    if( http_code >= 400 ) std::cout
        << "-NP_CURL::perform\n"
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

#ifdef WITHOUT_MAGIC
    dhdr.collect_json_content(download->buffer, download->size );
#endif

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

    NP* b = nullptr ;

#ifdef WITHOUT_MAGIC
    std::string dtype = dtype_convert::from_name(dhdr.dtype.c_str()) ;
    if(level > 0) std::cout << "-NP_CURL::collect_download dhdr.dtype[" << dhdr.dtype << "]\n" ;
    if(level > 0) std::cout << "-NP_CURL::collect_download      dtype[" << dtype << "]\n" ;

    b = new NP( dtype.c_str(), dhdr.sh );
    bool expect = b->uarr_bytes() == download->size ;

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

#else
    b = download ;
#endif

    if(level > 1) std::cout << desc();
    if(level > 0) std::cout << "]NP_CURL::collect_download\n" ;

    return b ;
}


inline std::string NP_CURL::desc() const
{
    std::stringstream ss ;
    ss
       << "[NP_CURL::desc\n"
#ifdef WITHOUT_MAGIC
       << " download " << download->desc() << "\n"
#else
       << " download " << ( download ? download->sstr() : "-" ) << "\n"
#endif
       << dhdr.desc() << "\n"
       << "]NP_CURL::desc\n"
       ;

    std::string str = ss.str() ;
    return str ;
}


