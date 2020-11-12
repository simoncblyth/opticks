
#include <cstdlib>
#include <iostream>
#include <boost/array.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#include "Viz.hh"
#include "ListenUDP.hh"

ListenUDP::ListenUDP(boost::asio::io_context& io_context, SCtrl* ctrl_ )
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
    std::cout 
        << "ListenUDP::ListenUDP"
        << " tid " << tid  
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


void ListenUDP::handle_receive_from(const boost::system::error_code& error,size_t bytes_recvd)
{
    std::string  message(data_, bytes_recvd); 

    std::string tid = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    std::cout 
        << "ListenUDP::handle_receive_from"
        << " tid " << tid  
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




