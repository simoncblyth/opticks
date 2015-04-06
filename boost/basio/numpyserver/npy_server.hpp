#pragma once

#include <vector>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <thread>

#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <asio-zmq.hpp>
#include "numpy.hpp"

#ifdef DEBUG
#include "dumpbuffer.hpp"
#endif

// following /usr/local/env/network/asiozmq/example/rrworker.cpp


template <class Delegate>
class npy_server : public boost::enable_shared_from_this<npy_server<Delegate>>  {

    boost::asio::zmq::socket             m_socket;
    std::vector<boost::asio::zmq::frame> m_buffer  ;
    Delegate*                            m_delegate ;
    boost::asio::io_service&             m_delegate_io_service ;

    std::string                          m_metadata ; 
    std::vector<int>                     m_shape ;
    std::vector<float>                   m_data ;
    bool                                 m_echo ;

    static const char JSON_MAGIC ; 
    static const char NPY_MAGIC ; 

public:
    npy_server(
                boost::asio::io_service&    io_service_, 
                Delegate*                   delegate,
                boost::asio::io_service&    delegate_io_service, 
                boost::asio::zmq::context&  zmq_ctx,
                const char*                 backend, 
                bool                        echo
              )
                : 
                   m_socket(io_service_, zmq_ctx, ZMQ_REP), 
                   m_buffer(),
                   m_delegate(delegate),
                   m_delegate_io_service(delegate_io_service),
                   m_echo(echo)
    {
#if VERBOSE
        std::cout 
             << std::setw(20) << boost::this_thread::get_id() 
             << " npy_server::npy_server " 
             << " backend " << backend 
             << std::endl;
#endif

        m_socket.connect(backend);   
        m_socket.async_read_message(
                     std::back_inserter(m_buffer),
                     std::bind(&npy_server::handle_request, this, std::placeholders::_1)
                 );
    }

    void response(std::vector<int> shape, std::vector<float> data,  std::string metadata );

private:
    void handle_request(boost::system::error_code const& ec);
    void echo() ;
    void decode_buffer();  // m_buffer -> m_metadata, m_shape and m_data 
    void decode_frame(char wanted);

private:
    // debug methods
    void dump();
    static void dump(std::vector<boost::asio::zmq::frame>& buffer);
    static void dump_npy( char* bytes, size_t size );
    void decode_buffer_roundtrip_test() ;
    void sleep(unsigned int secs);

private:
    std::vector<boost::asio::zmq::frame> make_frames(std::vector<int> shape, std::vector<float> data, std::string metadata);

};


template <typename Delegate>
const char npy_server<Delegate>::NPY_MAGIC = '\x93' ;

template <typename Delegate>
const char npy_server<Delegate>::JSON_MAGIC = '{' ;


template <typename Delegate>
void npy_server<Delegate>::sleep(unsigned int secs)
{
    std::this_thread::sleep_for(std::chrono::seconds(secs));
}

template <typename Delegate>
void npy_server<Delegate>::handle_request(boost::system::error_code const& ec)
{
#if VERBOSE
    std::cout 
             << std::setw(20) << boost::this_thread::get_id() 
             << " npy_server::handle_request  " 
             << std::endl;
#endif

    decode_buffer();
#ifdef DEBUG
    decode_buffer_roundtrip_test(); 
#endif

    // post decoded message to delegates on_npy handler
    m_delegate_io_service.post(
                        boost::bind(
                                &Delegate::on_npy,
                                m_delegate,
                                m_shape,
                                m_data,
                                m_metadata
                              ));



    // ZMQ demands a reply when using REQ/REP sockets 
    // so in non-echo mode the delegate must respond via "response" 
    // to avoid dropping the ZMQ REQ-REP ball
    //

    if(m_echo)
    {
        echo();
    }
    else
    {
    }


}


template <typename Delegate>
void npy_server<Delegate>::echo()
{

    m_socket.write_message(std::begin(m_buffer), std::end(m_buffer));

    // async call to handle_request when any subsequent requests arrive
    m_buffer.clear();
    m_socket.async_read_message(
                     std::back_inserter(m_buffer),
                     std::bind(&npy_server<Delegate>::handle_request, this, std::placeholders::_1)
                 );

}


template <typename Delegate>
void npy_server<Delegate>::response(std::vector<int> shape,std::vector<float> data,  std::string metadata )
{

#if VERBOSE
    std::cout 
             << std::setw(20) << boost::this_thread::get_id() 
             << " npy_server::response " 
             << std::endl;
#endif

    std::vector<boost::asio::zmq::frame> frames = make_frames(shape, data, metadata);

    // sync 
    m_socket.write_message(std::begin(frames), std::end(frames));

    // async call to handle_request when any subsequent requests arrive
    m_buffer.clear();
    m_socket.async_read_message(
                     std::back_inserter(m_buffer),
                     std::bind(&npy_server<Delegate>::handle_request, this, std::placeholders::_1)
                 );

}




template <typename Delegate>
void npy_server<Delegate>::decode_buffer()
{
    m_metadata.clear();
    m_shape.clear();
    m_data.clear();

    // populates m_metadata/m_shape/m_data
    decode_frame(JSON_MAGIC);
    decode_frame(NPY_MAGIC);

#if VERBOSE
    std::cout 
             << std::setw(20) << boost::this_thread::get_id() 
             << " npy_server::decode_buffer "
             << " metadata [" << m_metadata << "] "  
             << " shape dimensions " << m_shape.size() 
             << " data size " << m_data.size() 
             << std::endl;
#endif


}

template <typename Delegate>
void npy_server<Delegate>::decode_buffer_roundtrip_test()
{
    // re-serializes the deserialized m_shape/m_data/m_metadata 
    // and checks the frames are a bit-match for the original m_buffer 

    std::vector<boost::asio::zmq::frame> frames = make_frames(m_shape, m_data, m_metadata);

    std::cout << "npy_server::decode_buffer_roundtrip_test    frames.size() " <<   frames.size() << std::endl ; 
    std::cout << "npy_server::decode_buffer_roundtrip_test  m_buffer.size() " << m_buffer.size() << std::endl ; 

    assert(frames.size() == m_buffer.size());

    for(size_t i=0 ; i<frames.size() ; ++i)
    {
        const boost::asio::zmq::frame& a = m_buffer[i] ; 
        const boost::asio::zmq::frame& b = frames[i] ; 
        assert(a.size() == b.size());
        assert(memcmp( a.data(), b.data(), a.size()) == 0); 

#ifdef DEBUG
        if(*(char*)a.data() == NPY_MAGIC)
        {
            size_t offset = 0 ;   // 6*16 to skip header
            char* apt = (char*)a.data() + offset ;
            char* bpt = (char*)b.data() + offset ;
            int cmp = memcmp( apt, bpt, a.size()-offset );
            if(cmp)
            {
               DumpBuffer( (char*)a.data(), a.size(), 50);
               DumpBuffer( (char*)b.data(), b.size(), 50);
            } 
        }
#endif

    }

    //dump();
    //dump(frames);
 
}







template <typename Delegate>
std::vector<boost::asio::zmq::frame> npy_server<Delegate>::make_frames(std::vector<int> shape, std::vector<float> data, std::string metadata)
{
    // see G4DAEArray::SaveToBuffer()

    size_t dim = shape.size() ;
    std::stringstream ss ;

    int itemcount = shape[0] ;
    for(size_t i=1 ; i < dim ; ++i)
    {
        ss << shape[i] ;
        if( i < dim - 1 ) ss << ", " ;  // need the space for buffer memcmp matching
    } 
    std::string itemshape = ss.str();

    bool fortran_order = false ;

    // pre-calculate total buffer size including the padded header
    size_t nbytes = aoba::BufferSize<float>(shape[0], itemshape.c_str(), fortran_order  );

    // allocate frame to hold 
    boost::asio::zmq::frame npy_frame(nbytes);

    size_t wbytes = aoba::BufferSaveArrayAsNumpy<float>( (char*)npy_frame.data(), fortran_order, itemcount, itemshape.c_str(), data.data() );
    assert( wbytes == nbytes );

    std::vector<boost::asio::zmq::frame> frames ; 
    frames.push_back(npy_frame);
    frames.push_back(boost::asio::zmq::frame(metadata));

    return frames ; 
}




template <typename Delegate>
void npy_server<Delegate>::decode_frame(char wanted)
{
    for(unsigned int i=0 ; i < m_buffer.size() ; ++i )
    {
        const boost::asio::zmq::frame& frame = m_buffer[i] ; 
        char* bytes = (char*)frame.data();
        size_t size = frame.size(); 
        char peek = *bytes ;

        if(peek == wanted)
        {
            switch(peek)
            {
                case '{':
                      m_metadata.assign( bytes, bytes+size );
                      break;
                case '\x93':
                      aoba::BufferLoadArrayFromNumpy<float>(bytes, size, m_shape, m_data );
                      break;
                default:
                      break;
            }
        } 
        // last frame of each type wins, but only expecting one of each  
    }
}



template <typename Delegate>
void npy_server<Delegate>::dump()
{
    dump(m_buffer); 
}
 

template <typename Delegate>
void npy_server<Delegate>::dump(std::vector<boost::asio::zmq::frame>& buffer)
{
    for(unsigned int i=0 ; i < buffer.size() ; ++i )
    {
        const boost::asio::zmq::frame& frame = buffer[i] ; 
        char peek = *(char*)frame.data() ;
        printf("npy_server::dump frame %u/%lu size %8lu peek %c ", i, buffer.size(), frame.size(), peek );  
        if(peek == '{' )
        {
            printf(" JSON \n["); 
            fwrite((char*)frame.data(), sizeof(char), frame.size(), stdout);  // not null terminated
            printf("]\n");

            std::cout << std::to_string(frame) << "\n";
        } 
        else if(peek == '\x93')
        {
            printf(" NPY \n"); 
            dump_npy((char*)frame.data(), frame.size()); 
        }
        else
        {
            printf(" OTHER \n"); 
        }
    }
}


template <typename Delegate>
void npy_server<Delegate>::dump_npy( char* bytes, size_t size )
{
    // interpreting (bytes, size)  as serialized NPY array
    std::vector<int>  shape ;
    std::vector<float> data ;
    aoba::BufferLoadArrayFromNumpy<float>(bytes, size, shape, data );

    printf("npy_server::dump_npy data size %lu shape of %lu dimensions : ", data.size(), shape.size());
    int itemsize = 1 ;
    int fullsize = 1 ; 
    for(size_t d=0 ; d<shape.size(); ++d)
    {
       printf("%d ", shape[d]);
       if(d > 0) itemsize *= shape[d] ; 
       fullsize *= shape[d] ; 
    }
    int nitems = shape[0] ; 
    printf("\n itemsize %d fullsize %d nitems %d \n", itemsize, fullsize, nitems);
    assert(fullsize == data.size());


    for(size_t f=0 ; f<data.size(); ++f)
    {
         if(f < itemsize*3 || f >= (nitems - 3)*itemsize )
         {
             if(f % itemsize == 0) printf("%lu\n", f/itemsize);
             printf("%15.4f ", data[f]);
             if((f + 1) % itemsize == 0) printf("\n\n");
             else if((f+1) % 4 == 0) printf("\n");
         }
    }

}



