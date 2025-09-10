#pragma once

struct NP_CURL_Upload
{
    const char *data;
    size_t size;
    static size_t read_callback(void *buffer, size_t size, size_t nitems, void *userdata) ;
};

inline size_t NP_CURL_Upload::read_callback(void *buffer, size_t size, size_t nitems, void *userdata)
{
    struct NP_CURL_Upload* upload = (struct NP_CURL_Upload *)userdata;
    size_t copy_size = size * nitems;

    if (copy_size > upload->size) copy_size = upload->size; // for buffered read make sure to stay in range

    memcpy(buffer, upload->data, copy_size);

    upload->data += copy_size;  // move data pointer
    upload->size -= copy_size;  // decrease remaining size

    return copy_size;
}


