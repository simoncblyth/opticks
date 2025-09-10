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

inline size_t NP_CURL_Download::write_callback(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    struct NP_CURL_Download* download = (struct NP_CURL_Download *)userdata;
    size_t new_len = download->size + (size * nmemb);
    char *new_buffer = (char*)realloc(download->buffer, new_len + 1);

    if (new_buffer == NULL) {
        // Realloc failed, a real-world app would handle this more robustly
        fprintf(stderr, "realloc() failed!\n");
        return 0; // Abort transfer
    }

    // Update the buffer pointer and size
    download->buffer = new_buffer;
    memcpy(&(download->buffer[download->size]), ptr, size * nmemb);
    download->buffer[new_len] = '\0';
    download->size = new_len;

    return size * nmemb;
}



