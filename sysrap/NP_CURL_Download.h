#pragma once


struct NP_CURL_Download {
    char *buffer;
    size_t size;

    void clear();
    std::string desc() const ;

    static size_t write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) ;
};

inline void NP_CURL_Download::clear()
{
    free(buffer);
    buffer = nullptr ;
}

inline std::string NP_CURL_Download::desc() const
{
    std::stringstream ss ;
    ss << "NP_CURL_Download::desc " << size << "\n" ;
    std::string str = ss.str() ;
    return str ;
}

/**
NP_CURL_Download::write_callback
-----------------------------------

This is used from NP_CURL::prepare_download
which starts from download->size zero and one bytes buffer.

This callback is called multiple times with non-zero size*nmemb
data from the ptr is copied to the download->buffer

HMM is content length always available

1. extend download->buffer size to fit arriving data and update download->buffer pointer as may have been moved by realloc
2. copy arriving bytes into the buffer using download->size as offset and update the offset
3. return the number of bytes written to the download->buffer

**/


inline size_t NP_CURL_Download::write_callback(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    struct NP_CURL_Download* download = (struct NP_CURL_Download *)userdata;

    // 1. extend download->buffer size to fit arriving data and update download->buffer pointer as may have been moved by realloc
    size_t new_len = download->size + (size * nmemb);
    char *new_buffer = (char*)realloc(download->buffer, new_len + 1);
    if(new_buffer == nullptr) fprintf(stderr, "NP_CURL_Download::write_callback realloc() failed!\n");
    if(new_buffer == nullptr) return 0; // Abort transfer
    download->buffer = new_buffer;

    // 2. copy arriving bytes into the buffer using download->size as offset and update the offset
    memcpy(&(download->buffer[download->size]), ptr, size * nmemb);
    download->buffer[new_len] = '\0';
    download->size = new_len;

    // 3. return the number of bytes written to the download->buffer
    return size * nmemb;
}



