#pragma once

#include <boost/asio.hpp>
#include "plog/Severity.h"
#include "BRAP_API_EXPORT.hh"

/**
BListenUDP
===========

This UDP listener means that messages sent to a running OpticksViz GUI 
can control it, changing viewpoint etc.. by parsing a subset of the launch 
commandline arguments. 

Despite the appearance of concurrency all the BListenUDP code runs on 
the main thread : that is the magic of boost::asio and the async io approach.

See oglrap/OpticksViz.cc for canonical usage with m_listen_udp::

    103 OpticksViz::OpticksViz(OpticksHub* hub, OpticksIdx* idx, bool immediate)
    104     :
    ..
    108 #ifdef WITH_BOOST_ASIO
    109     m_io_context(),
    110     m_listen_udp(new BListenUDP<OpticksViz>(m_io_context,this)),   // UDP messages -> ::command calls to this
    111 #endif

    ...   // GUI renderloop
 
    576     while (!glfwWindowShouldClose(m_window) && !exitloop  )
    577     {
    ...
    583 #ifdef WITH_BOOST_ASIO
    584         m_io_context.poll_one();
    585 #endif

**/


template <class T>
class BRAP_API BListenUDP
{
    private:
        static const plog::Severity LEVEL ; 
    public:
        BListenUDP(boost::asio::io_context& io_context, T* ctrl_);
    private:
        void handle_receive_from(const boost::system::error_code& error,size_t bytes_recvd);
    private:
        T*          ctrl ; 
        const char* host_ ; 
        const char* port_ ; 
        boost::asio::io_context& io_context_ ; 
        boost::asio::ip::address host ;
        const int port ;
        boost::asio::ip::udp::endpoint local_endpoint  ;  
        boost::asio::ip::udp::endpoint sender_endpoint ;
        boost::asio::ip::udp::socket   socket ;
    private:
        enum {  max_length = 1024 } ; 
        char  data_[max_length] ; 
};



#include <cstdlib>
#include <iostream>
#include <boost/array.hpp>
#include <boost/thread.hpp>    // just to report thread id : no threads are created 
#include <boost/lexical_cast.hpp>

#include "PLOG.hh"

template <typename T>
const plog::Severity BListenUDP<T>::LEVEL = PLOG::EnvLevel("BListenUDP", "DEBUG") ; 

/**
BListenUDP::BListenUDP
--------------------------

Bind to the UDP socket and setup a callback for when UDP messages are received.

**/

template <typename T>
BListenUDP<T>::BListenUDP(boost::asio::io_context& io_context, T* ctrl_ )
    :
    ctrl(ctrl_),
    host_(getenv("UDP_HOST")),
    port_(getenv("UDP_PORT")),
    io_context_(io_context),
    host(boost::asio::ip::address::from_string(host_ ? host_ : "127.0.0.1")),
    port(boost::lexical_cast<int>(port_ ? port_ : "15001")),
    local_endpoint(boost::asio::ip::udp::endpoint( host, port )),
    socket(io_context_)
{
    std::string tid = boost::lexical_cast<std::string>(boost::this_thread::get_id());

    LOG(LEVEL) 
        << " tid " << tid  
        << " local_endpoint " << local_endpoint 
        ;

    LOG(info) 
        << " tid " << tid  
        << " local_endpoint " << local_endpoint 
        ;



    assert( ctrl ); 

    socket.open(boost::asio::ip::udp::v4());
    socket.bind(local_endpoint);
    
    socket.async_receive_from(
        boost::asio::buffer(data_, max_length), sender_endpoint,
        boost::bind(&BListenUDP::handle_receive_from, this, boost::asio::placeholders::error,boost::asio::placeholders::bytes_transferred)
    );
}


/**
BListenUDP::handle_receive_from
----------------------------------

1. the message receive is passed to ctrl delegate with ctrl->command( message )
2. a further async callback is setup to receive the next message  

NB for this to run it is necessary to poll the io_context from the GUI render loop

**/

template <typename T>
void BListenUDP<T>::handle_receive_from(const boost::system::error_code& error,size_t bytes_recvd)
{
    std::string  message(data_, bytes_recvd); 

    std::string tid = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    std::cout 
        << "BListenUDP::handle_receive_from"
        << " tid " << tid  
        << " bytes_recvd " << bytes_recvd 
        << " message " << message  
        << std::endl
        ;

    ctrl->command( strdup(message.c_str()) );  

    socket.async_receive_from(
        boost::asio::buffer(data_, max_length), sender_endpoint,
        boost::bind(&BListenUDP::handle_receive_from, this, boost::asio::placeholders::error,boost::asio::placeholders::bytes_transferred)
    );
}


