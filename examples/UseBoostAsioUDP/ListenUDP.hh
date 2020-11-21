#pragma once

#include <boost/asio.hpp>
#define API  __attribute__ ((visibility ("default")))

template <class T>
class API ListenUDP
{
    public:
        ListenUDP(boost::asio::io_context& io_context, T* ctrl_);
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
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>

#ifdef WITH_BOOST_THREAD
#include <boost/thread.hpp>
std::string TID()
{
    std::string tid = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    return tid ;  
}
#else
std::string TID()
{
    return "-" ; 
}
#endif


template <typename T>
ListenUDP<T>::ListenUDP(boost::asio::io_context& io_context, T* ctrl_ )
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
    std::cout 
        << "ListenUDP::ListenUDP"
        << " tid " << TID()
        << " local_endpoint " << local_endpoint 
        << std::endl
        ;

    assert( ctrl ); 

    socket.open(boost::asio::ip::udp::v4());
    socket.bind(local_endpoint);
    
    socket.async_receive_from(
        boost::asio::buffer(data_, max_length), sender_endpoint,
        boost::bind(&ListenUDP::handle_receive_from, this, boost::asio::placeholders::error,boost::asio::placeholders::bytes_transferred)
    );
}


template <typename T>
void ListenUDP<T>::handle_receive_from(const boost::system::error_code& error,size_t bytes_recvd)
{
    std::string  message(data_, bytes_recvd); 

    std::cout 
        << "ListenUDP::handle_receive_from"
        << " tid " << TID() 
        << " bytes_recvd " << bytes_recvd 
        << " message " << message  
        << std::endl
        ;

    ctrl->command( strdup(message.c_str()) );  

    socket.async_receive_from(
        boost::asio::buffer(data_, max_length), sender_endpoint,
        boost::bind(&ListenUDP::handle_receive_from, this, boost::asio::placeholders::error,boost::asio::placeholders::bytes_transferred)
    );
}


