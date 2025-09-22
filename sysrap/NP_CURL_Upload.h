#pragma once



struct NP_CURL_Upload_0
{
    char* buffer;
    size_t size;
    size_t position ; // transitional to match _1

    static size_t read_callback(void *buffer, size_t size, size_t nitems, void *userdata) ;
};

inline size_t NP_CURL_Upload_0::read_callback(void *buffer, size_t size, size_t nitems, void *userdata)
{
    struct NP_CURL_Upload_0* upload = (struct NP_CURL_Upload_0 *)userdata;
    size_t copy_size = size * nitems;

    if (copy_size > upload->size) copy_size = upload->size; // for buffered read make sure to stay in range

    memcpy(buffer, upload->buffer, copy_size);

    upload->buffer += copy_size;  // move buffer pointer
    upload->size   -= copy_size;  // decrease remaining size

    return copy_size;
}


/**

https://curl.se/libcurl/c/curl_mime_data_cb.html

**/


struct NP_CURL_Upload_1
{
    char* buffer ;
    curl_off_t size;
    curl_off_t position ;

    std::string desc() const ;

    static size_t read_callback(char *buffer, size_t size, size_t nitems, void* arg) ;
};


inline std::string NP_CURL_Upload_1::desc() const
{
    std::stringstream ss ;
    ss << "-NP_CURL_Upload_1 size " << size << " position " << position << "\n" ;
    std::string str = ss.str() ;
    return str ;
}


inline size_t NP_CURL_Upload_1::read_callback(char* buffer, size_t size, size_t nitems, void* arg)
{
    curl_off_t copy_sz = size*nitems ;

    struct NP_CURL_Upload_1* p = (struct NP_CURL_Upload_1 *)arg;

    curl_off_t sz = p->size - p->position; // remaining

    if (sz > copy_sz) sz = copy_sz ; // for buffered read make sure to stay in range

    if(false) std::cout
        << "[NP_CURL_Upload_1::read_callback"
        << " size " << size
        << " nitems " << nitems
        << " copy_sz " << copy_sz
        << " p.desc " << p->desc()
        << " sz " << sz
        << "\n"
        ;


    if(sz) memcpy(buffer, p->buffer + p->position, sz);

    p->position += sz ;
    // not moving the buffer pointer unlike above

    if(false) std::cout
        << "]NP_CURL_Upload_1::read_callback"
        << " size " << size
        << " nitems " << nitems
        << "\n"
        ;


    return sz ;
}









