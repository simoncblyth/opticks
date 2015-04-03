#pragma once

#include <vector>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <thread>

#include <boost/asio.hpp>
#include <asio-zmq.hpp>
#include "numpy.hpp"

// following /usr/local/env/network/asiozmq/example/rrworker.cpp

template <class Delegate>
class npy_server {

    //boost::asio::zmq::context&           m_ctx;
    boost::asio::zmq::socket             m_socket;
    std::vector<boost::asio::zmq::frame> m_buffer  ;
    Delegate*                            m_delegate ;
    boost::asio::io_service&             m_delegate_io_service ;

    std::string                          m_metadata ; 
    std::vector<int>                     m_shape ;
    std::vector<float>                   m_data ;

public:
    npy_server(
                boost::asio::zmq::context&  zmq_ctx,
                boost::asio::io_service&    io_service_, 
                Delegate*                   delegate,
                boost::asio::io_service&    delegate_io_service, 
                const char*                 backend
              )
                : 
                   m_socket(io_service_, zmq_ctx, ZMQ_REP), 
                   m_buffer(),
                   m_delegate(delegate),
                   m_delegate_io_service(delegate_io_service)
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
private:
    void handle_request(boost::system::error_code const& ec);

private:
    void dump();
    void dump_npy( char* bytes, size_t size );
    void decode_buffer();
    void decode_frame(char wanted);
    void sleep(unsigned int secs);

};

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
    m_delegate_io_service.post(
                        boost::bind(
                                &Delegate::on_npy,
                                m_delegate,
                                m_shape,
                                m_data,
                                m_metadata
                              ));


    m_socket.write_message(std::begin(m_buffer), std::end(m_buffer));

    // Currently just echoing : ZMQ demands a reply when using REQ/REP sockets 
    //
    // If need processing elsewhere (eg GPU thread) 
    // would need to split into
    //
    // handle_request 
    //       decode + post to delegate
    //
    // send_reply(processed_result msg)
    //       the delegate (or another actor maybe GPU thread) 
    //       needs to post to this as a result of receiving the on_npy
    //       in order to send results back to the requester
    //       then keep ball rolling by async_read_message with handle_request
    // 
    //
    // async call with this method as handler, to pickup subsequent requests
    // 

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

    decode_frame('{');
    decode_frame('\x93');

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
    for(unsigned int i=0 ; i < m_buffer.size() ; ++i )
    {
        const boost::asio::zmq::frame& frame = m_buffer[i] ; 
        char peek = *(char*)frame.data() ;
        printf("npy_server::dump frame %u/%lu size %8lu peek %c ", i, m_buffer.size(), frame.size(), peek );  
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



