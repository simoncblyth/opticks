#pragma once


struct NP_CURL_Header
{
    typedef std::int64_t INT ;


    std::string name ;

    std::string token ;
    int         level ;
    int         index ;

    std::string dtype ;
    std::string shape ;
    std::vector<INT> sh ;

    struct curl_slist* headerlist ;


    size_t      c_length ;
    std::string c_type ;
    std::string content ; // usually empty, only populated after HTTP error with json response



    NP_CURL_Header(const char* name);

    void prepare_upload(const char* token_, int index_, int level_=0, const char* dtype_=nullptr, const char* shape_=nullptr );
    void clear();
    void collect( const char* name, const char* value );
    void collect_json_content( char* buffer, size_t size );
    std::string sstr() const ;
    std::string desc() const ;

    static constexpr const char* x_opticks_token = "x-opticks-token" ;
    static constexpr const char* x_opticks_level = "x-opticks-level" ; // debug level integer
    static constexpr const char* x_opticks_index = "x-opticks-index" ;

    static constexpr const char* x_opticks_dtype = "x-opticks-dtype" ;
    static constexpr const char* x_opticks_shape = "x-opticks-shape" ;

    static constexpr const char* content_length = "content-length" ;
    static constexpr const char* content_type   = "content-type" ;

    static  std::string Format(const char* prefix, int value);
    static  std::string Format_LEVEL(int level);
    static  std::string Format_INDEX(int index);

    static  std::string Format(const char* prefix, const char* value);
    static  std::string Format_TOKEN(const char* token);
    static  std::string Format_DTYPE(const char* dtype);
    static  std::string Format_SHAPE(const char* shape);

    static void Parse_SHAPE( std::vector<INT>& sh, const char* shape );
    static bool Expected_DTYPE(const char* dtype);
};

inline NP_CURL_Header::NP_CURL_Header( const char* name_ )
    :
    name(name_),
    level(0),
    index(0),
    headerlist(nullptr)
{
}

inline void NP_CURL_Header::prepare_upload(const char* token_, int index_, int level_, const char* dtype_, const char* shape_ )
{
    std::string x_index = Format_INDEX(index_) ;
    std::string x_level = Format_LEVEL(level_) ;

    assert( headerlist == nullptr );  // should have been cleared
    headerlist = curl_slist_append(headerlist, x_index.c_str() );
    headerlist = curl_slist_append(headerlist, x_level.c_str() );


    if( token_ )
    {
        std::string x_token = Format_TOKEN(token_) ;
        headerlist = curl_slist_append(headerlist, x_token.c_str() );
    }

    if( dtype_ )
    {
        bool expected_dtype = Expected_DTYPE(dtype_);
        assert( expected_dtype );
        std::string x_dtype = Format_DTYPE(dtype_) ;
        headerlist = curl_slist_append(headerlist, x_dtype.c_str() );
    }

    if( shape_ )
    {
        std::string x_shape = Format_SHAPE(shape_) ;
        headerlist = curl_slist_append(headerlist, x_shape.c_str() );
    }
}

inline void NP_CURL_Header::clear()  // clears everything other than name
{
    token.clear();
    level = 0 ;
    index = 0 ;
    dtype.clear();
    shape.clear();
    sh.clear();
    c_length = 0 ;
    c_type.clear();
    content.clear();

    curl_slist_free_all(headerlist);
    headerlist = nullptr ;
}

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
    else if( 0==strcmp(name,x_opticks_token))
    {
        token = value ;
    }
    else if( 0==strcmp(name,x_opticks_dtype))
    {
        dtype = value ;
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
    ss << std::setw(20) << x_opticks_dtype << " : " << dtype << "\n" ;
    ss << std::setw(20) << x_opticks_shape << " : " << shape << "\n" ;
    ss << std::setw(20) << "sh.size"     << " : " << sh.size() << "\n" ;
    ss << std::setw(20) << "sstr"        << " : " << sstr() << "\n" ;
    ss << std::setw(20) << content_length << " : " << c_length << "\n" ;
    ss << std::setw(20) << content_type   << " : " << c_type << "\n" ;
    ss << std::setw(20) << "json_content" << "\n" << content << "\n" ;

    ss << "]NP_CURL_Header::desc\n" ;
    std::string str = ss.str();
    return str ;
}



inline std::string NP_CURL_Header::Format( const char* prefix, int value )
{
    std::stringstream ss ;
    ss << prefix << ":" << value ;
    std::string str = ss.str();
    return str ;
}
inline std::string NP_CURL_Header::Format_LEVEL( int level ){  return Format(x_opticks_level, level ); }
inline std::string NP_CURL_Header::Format_INDEX( int index ){  return Format(x_opticks_index, index ); }


inline std::string NP_CURL_Header::Format( const char* prefix, const char* value )
{
    std::stringstream ss ;
    ss << prefix << ":" << value ;
    std::string str = ss.str();
    return str ;
}

inline std::string NP_CURL_Header::Format_TOKEN( const char* token ){ return Format(x_opticks_token, token ); }
inline std::string NP_CURL_Header::Format_DTYPE( const char* dtype ){ return Format(x_opticks_dtype, dtype ); }
inline std::string NP_CURL_Header::Format_SHAPE( const char* shape ){ return Format(x_opticks_shape, shape ); }

inline void NP_CURL_Header::Parse_SHAPE( std::vector<INT>& sh, const char* shape )
{
    INT num;
    std::stringstream ss;
    for (int i=0 ; i < int(strlen(shape)) ; i++) ss << (std::isdigit(shape[i]) ? shape[i] : ' ' ) ; // replace non-digits with spaces
    while (ss >> num) sh.push_back(num);
}

inline bool NP_CURL_Header::Expected_DTYPE(const char* dtype)
{
     // TODO: move this into NPU.hh
     return    strcmp(dtype, "float32") == 0
            || strcmp(dtype, "float64") == 0
            || strcmp(dtype, "int64")   == 0
            || strcmp(dtype, "int32")   == 0
            || strcmp(dtype, "int16")   == 0
            || strcmp(dtype, "int8")    == 0
            || strcmp(dtype, "uint64")  == 0
            || strcmp(dtype, "uint32")  == 0
            || strcmp(dtype, "uint16")  == 0
            || strcmp(dtype, "uint8")   == 0
               ;
}




