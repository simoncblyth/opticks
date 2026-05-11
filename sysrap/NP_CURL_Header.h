#pragma once

#include <vector>
#include <string>
#include <cstring>

struct NP_CURL_Header
{
    static std::string URL_Encode(const std::string& value);
    typedef std::int64_t INT ;


    std::string name ;

    std::string token ;
    size_t      level ;
    size_t      index ;
    size_t      count ;
    std::string meta  ;

    std::string dtype ;
    std::string shape ;
    std::vector<INT> sh ;

    struct curl_slist* headerlist ;

    std::vector<std::string> response_header_lines ; // raw header lines from response
    std::string header_buffer ; // buffer for partial header lines in callback


    size_t      c_length ;
    std::string c_type ;
    std::string content ; // usually empty, only populated after HTTP error with json response



    NP_CURL_Header(const char* name);

    void prepare_upload(
         const std::string& token_,
         size_t index_,
         size_t level_,
         size_t count_,
         const std::string& dtype_,
         const std::string& shape_,
         const std::string& meta_ );

    void clear();
    void collect( const char* name, const char* value );


    void collect_header_bytes_process(); // Helper for header line buffering (called by collect_header_bytes)
    void collect_header_bytes( const char* chunk, size_t len );  // collect raw header chunk for fallback
    // CURLOPT_HEADERFUNCTION callback used for libcurl < 8.12.1
    static size_t CollectHeaderBytes(char* buffer, size_t size, size_t nitems, void* userdata);
    void parse_response_headers();  // parse collected header lines into name/value pairs


    void collect_json_content( char* buffer, size_t size );
    std::string sstr() const ;
    std::string desc() const ;


    static constexpr const char* x_opticks_token = "x-opticks-token" ;
    static constexpr const char* x_opticks_index = "x-opticks-index" ;  // eg EventID
    static constexpr const char* x_opticks_level = "x-opticks-level" ;  // debug level integer
    static constexpr const char* x_opticks_count = "x-opticks-count" ;  // eg NumPhoton

    static constexpr const char* x_opticks_dtype = "x-opticks-dtype" ;
    static constexpr const char* x_opticks_shape = "x-opticks-shape" ;
    static constexpr const char* x_opticks_meta = "x-opticks-meta" ;    // general metadata

    static constexpr const char* content_length = "content-length" ;
    static constexpr const char* content_type   = "content-type" ;

    static  std::string Format(const char* prefix, size_t value);
    static  std::string Format_INDEX(size_t index);
    static  std::string Format_LEVEL(size_t level);
    static  std::string Format_COUNT(size_t count);

    static  std::string FormatString(const char* prefix, const std::string& value);
    static  std::string Format_TOKEN(const std::string& token);
    static  std::string Format_DTYPE(const std::string& dtype);
    static  std::string Format_SHAPE(const std::string& shape);
    static  std::string Format_META(const std::string& meta);

    static void Parse_SHAPE( std::vector<INT>& sh, const char* shape );
    static bool Expected_DTYPE(const std::string& dtype);

};




#include <iostream>
#include <sstream>
#include <iomanip>


inline std::string NP_CURL_Header::URL_Encode(const std::string& value)
{
    std::ostringstream escaped;
    escaped << std::hex << std::uppercase;

    for (char c : value) {
        // Keep alphanumeric and other safe characters
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
        } else {
            // Any other character (including \n) becomes %XX
            escaped << '%' << std::setw(2) << std::setfill('0') << (int)(unsigned char)c;
        }
    }
    return escaped.str();
}








inline NP_CURL_Header::NP_CURL_Header( const char* name_ )
    :
    name(name_),
    level(0),
    index(0),
    count(0),
    headerlist(nullptr)
{
}

/**
NP_CURL_Header::prepare_upload
--------------------------------

Populates headerlist with the non-nullptr arguments

**/


inline void NP_CURL_Header::prepare_upload(const std::string& token_, size_t index_, size_t level_, size_t count_, const std::string& dtype_, const std::string& shape_, const std::string& meta_ )
{
    bool expected_dtype = Expected_DTYPE(dtype_);
    assert( expected_dtype );

    std::string x_level = Format_LEVEL(level_) ;
    std::string x_index = Format_INDEX(index_) ;
    std::string x_count = Format_COUNT(count_) ;

    std::string x_token = Format_TOKEN(token_) ;
    std::string x_dtype = Format_DTYPE(dtype_) ;
    std::string x_shape = Format_SHAPE(shape_) ;
    std::string x_meta  = Format_META(meta_) ;

    assert( headerlist == nullptr );  // should have been cleared
    headerlist = curl_slist_append(headerlist, x_index.c_str() );
    headerlist = curl_slist_append(headerlist, x_level.c_str() );
    headerlist = curl_slist_append(headerlist, x_count.c_str() );

    headerlist = curl_slist_append(headerlist, x_token.c_str() );
    headerlist = curl_slist_append(headerlist, x_dtype.c_str() );
    headerlist = curl_slist_append(headerlist, x_shape.c_str() );
    headerlist = curl_slist_append(headerlist, x_meta.c_str() );
}

inline void NP_CURL_Header::clear()  // clears everything other than name
{
    token.clear();
    level = 0 ;
    index = 0 ;
    count = 0 ;
    dtype.clear();
    shape.clear();
    meta.clear();
    sh.clear();
    c_length = 0 ;
    c_type.clear();
    content.clear();
    response_header_lines.clear();
    header_buffer.clear();

    curl_slist_free_all(headerlist);
    headerlist = nullptr ;
}

/**
NP_CURL_Header::collect
-------------------------

Sets the below based on the field name provides::

    level
    index
    token
    dtype
    shape and sh vector
    c_length
    c_type

**/


inline void NP_CURL_Header::collect( const char* name, const char* value )
{
    if( 0==strcmp(name,x_opticks_level))
    {
        std::stringstream ss(value);
        ss >> level ;
    }
    else if( 0==strcmp(name,x_opticks_index))
    {
        std::stringstream ss(value);
        ss >> index ;
    }
    else if( 0==strcmp(name,x_opticks_count))
    {
        std::stringstream ss(value);
        ss >> count ;
    }
    else if( 0==strcmp(name,x_opticks_token))
    {
        token = value ;
    }
    else if( 0==strcmp(name,x_opticks_dtype))
    {
        dtype = value ;
    }
    else if( 0==strcmp(name,x_opticks_meta))
    {
        meta = value ;
    }
    else if( 0==strcmp(name,x_opticks_shape))
    {
        shape = value ;
        Parse_SHAPE(sh, shape.c_str());
    }
    else if( 0==strcmp(name,content_length))
    {
        std::stringstream ss(value);
        ss >> c_length ;
    }
    else if( 0==strcmp(name,content_type))
    {
         c_type = value ;
    }
}

/**
NP_CURL_Header::collect_header_bytes_process
---------------------------------------------

Helper to process complete header lines from the buffer.
Handles CRLF, CR, and LF line endings.

1. while header_buffer bytes find eol and collect lines into response_header_lines
2. Keep any leftover partial line in header_buffer


**/
inline void NP_CURL_Header::collect_header_bytes_process()
{
    // 1. while header_buffer bytes find eol and collect lines into response_header_lines

    size_t pos = 0;
    while(pos < header_buffer.size())
    {
        // Find line ending: CRLF, CR, or LF
        size_t eol = std::string::npos;
        size_t crlf = header_buffer.find("\r\n", pos);
        size_t cr = header_buffer.find('\r', pos);
        size_t lf = header_buffer.find('\n', pos);

        if(crlf != std::string::npos && crlf == std::min(crlf, std::min(cr, lf)))
        {
            eol = crlf;
        }
        else if(cr != std::string::npos && (lf == std::string::npos || cr < lf))
        {
            eol = cr;
        }
        else if(lf != std::string::npos)
        {
            eol = lf;
        }

        if(eol == std::string::npos) break;

        if(eol > pos)
        {
            std::string line = header_buffer.substr(pos, eol - pos);
            if(!line.empty())
            {
                response_header_lines.push_back(line);
            }
        }

        // Skip line ending
        if(eol + 1 < header_buffer.size() && header_buffer[eol] == '\r' && header_buffer[eol+1] == '\n')
        {
            pos = eol + 2;
        }
        else
        {
            pos = eol + 1;
        }
    }


    // 2. Keep any leftover partial line in buffer

    if(pos > 0 && pos < header_buffer.size())
    {
        header_buffer = header_buffer.substr(pos);
    }
    else
    {
        header_buffer.clear();
    }
}

inline void NP_CURL_Header::collect_header_bytes( const char* chunk, size_t len )
{
    if(!chunk || len == 0) return;

    header_buffer.append(chunk, len);
    collect_header_bytes_process();
}

/**
NP_CURL_Header::CollectHeaderBytes
------------------------------------

Used from NP_CURL::prepare_download for older libcurl without NP_CURL_HAVE_NEXTHEADER

**/


inline size_t NP_CURL_Header::CollectHeaderBytes(char* buffer, size_t size, size_t nitems, void* userdata)
{
    size_t realsize = size * nitems;
    NP_CURL_Header* hdr = static_cast<NP_CURL_Header*>(userdata);
    if(hdr && realsize > 0)
    {
        hdr->collect_header_bytes(buffer, realsize);
    }
    return realsize;
}



/**
NP_CURL_Header::parse_response_headers
--------------------------------------

For older libcurl without NP_CURL_HAVE_NEXTHEADER this
collects header key value pairs from the response_header_lines
collected by the above NP_CURL_Header::CollectHeaderBytes callback.
This is invoked at tail of NP_CURL::perform

**/

inline void NP_CURL_Header::parse_response_headers()
{
    // Flush any remaining data in buffer (should be empty at end of headers)
    if(!header_buffer.empty())
    {
        // Treat remaining as a line (might be missing final newline)
        if(!header_buffer.empty())
        {
            response_header_lines.push_back(header_buffer);
        }
        header_buffer.clear();
    }

    for(const std::string& line : response_header_lines)
    {
        const char* l = line.c_str();
        const char* colon = strchr(l, ':');
        if(colon && colon > l)
        {
            std::string name(l, colon - l);
            const char* value = colon + 1;
            while(*value == ' ') value++;  // skip leading whitespace
            collect(name.c_str(), value);
        }
    }
    response_header_lines.clear();
}


inline void NP_CURL_Header::collect_json_content( char* buffer, size_t size )
{
    if( c_type == "application/json" && c_length > 0 && c_length == size )
    {
        content.resize( size + 1  );  // +1 ?
        memcpy(content.data(), buffer, size ) ;
        content.data()[size] = '\0' ;
    }
}


inline std::string NP_CURL_Header::sstr() const
{
    int num = sh.size();
    std::stringstream ss ;
    for(int i=0 ; i < num ; i++) ss << sh[i] << ( i < num - 1 ? "," : " " ) ;
    std::string str = ss.str() ;
    return str ;
}


inline std::string NP_CURL_Header::desc() const
{
    std::stringstream ss ;
    ss << "[NP_CURL_Header::desc [" << name << "]\n" ;
    ss << std::setw(20) << x_opticks_token << " : " << token << "\n" ;
    ss << std::setw(20) << x_opticks_level << " : " << level << "\n" ;
    ss << std::setw(20) << x_opticks_index << " : " << index << "\n" ;
    ss << std::setw(20) << x_opticks_count << " : " << count << "\n" ;
    ss << std::setw(20) << x_opticks_dtype << " : " << dtype << "\n" ;
    ss << std::setw(20) << x_opticks_shape << " : " << shape << "\n" ;
    ss << std::setw(20) << "sh.size"     << " : " << sh.size() << "\n" ;
    ss << std::setw(20) << x_opticks_meta  << " : " << meta << "\n" ;
    ss << std::setw(20) << "sstr"        << " : " << sstr() << "\n" ;
    ss << std::setw(20) << content_length << " : " << c_length << "\n" ;
    ss << std::setw(20) << content_type   << " : " << c_type << "\n" ;
    ss << std::setw(20) << "json_content" << "\n" << content << "\n" ;

    ss << "]NP_CURL_Header::desc\n" ;
    std::string str = ss.str();
    return str ;
}



inline std::string NP_CURL_Header::Format( const char* prefix, size_t value )
{
    std::stringstream ss ;
    ss << prefix << ":" << value ;
    std::string str = ss.str();
    return str ;
}
inline std::string NP_CURL_Header::Format_LEVEL( size_t level ){  return Format(x_opticks_level, level ); }
inline std::string NP_CURL_Header::Format_INDEX( size_t index ){  return Format(x_opticks_index, index ); }
inline std::string NP_CURL_Header::Format_COUNT( size_t count ){  return Format(x_opticks_count, count ); }


inline std::string NP_CURL_Header::FormatString( const char* prefix, const std::string& value )
{
    std::stringstream ss ;
    ss << prefix << ":" << value ;
    std::string str = ss.str();
    return str ;
}

inline std::string NP_CURL_Header::Format_TOKEN( const std::string& token ){ return FormatString(x_opticks_token, token ); }
inline std::string NP_CURL_Header::Format_DTYPE( const std::string& dtype ){ return FormatString(x_opticks_dtype, dtype ); }
inline std::string NP_CURL_Header::Format_SHAPE( const std::string& shape ){ return FormatString(x_opticks_shape, shape ); }
inline std::string NP_CURL_Header::Format_META(  const std::string& meta_ )
{
    std::string meta = URL_Encode(meta_);
    return FormatString(x_opticks_meta,  meta.c_str()  );
}

inline void NP_CURL_Header::Parse_SHAPE( std::vector<INT>& sh, const char* shape )
{
    INT num;
    std::stringstream ss;
    for (int i=0 ; i < int(strlen(shape)) ; i++) ss << (std::isdigit(shape[i]) ? shape[i] : ' ' ) ; // replace non-digits with spaces
    while (ss >> num) sh.push_back(num);
}

inline bool NP_CURL_Header::Expected_DTYPE(const std::string& dtype)
{
     return dtype_convert::expected_noncomplex_dtype(dtype.c_str());
}


