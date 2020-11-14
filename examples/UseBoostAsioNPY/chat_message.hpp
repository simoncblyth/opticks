// /usr/local/env/boost/basio/example/cpp03/chat/chat_message.hpp
// chat_message.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef CHAT_MESSAGE_HPP
#define CHAT_MESSAGE_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>

class chat_message
{
    public:
        enum { header_length = 4 };
        enum { max_body_length = 512 };

        chat_message();

        const char* data() const ;
        char*       data();
        size_t      length() const ;
        const char* body() const ;
        char*       body() ;
        size_t      body_length() const ;
        void        body_length(size_t new_length);
        bool        decode_header();
        void        encode_header();
    private:
        char        data_[header_length + max_body_length];
        size_t      body_length_;
};

chat_message::chat_message()
    : 
    body_length_(0)
{
}
const char* chat_message::data() const
{
    return data_;
}
char* chat_message::data()
{
    return data_;
}
size_t chat_message::length() const
{
    return header_length + body_length_;
}
const char* chat_message::body() const
{
    return data_ + header_length;
}
char* chat_message::body()
{
    return data_ + header_length;
}
size_t chat_message::body_length() const
{
    return body_length_;
}
void chat_message::body_length(size_t new_length)
{
    body_length_ = new_length;
    if (body_length_ > max_body_length) body_length_ = max_body_length;
}
bool chat_message::decode_header()
{
    using namespace std; // For strncat and atoi.
    char header[header_length + 1] = "";
    strncat(header, data_, header_length);
    body_length_ = atoi(header);
    if (body_length_ > max_body_length)
    {
        body_length_ = 0;
        return false;
    }
    return true;
}
void chat_message::encode_header()
{
    using namespace std; // For sprintf and memcpy.
    char header[header_length + 1] = "";
    sprintf(header, "%4d", static_cast<int>(body_length_));
    memcpy(data_, header, header_length);
}

#endif // CHAT_MESSAGE_HPP
