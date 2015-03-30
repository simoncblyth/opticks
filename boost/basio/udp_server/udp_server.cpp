// based on /usr/local/env/boost/basio/example/cpp03/tutorial/daytime6/server.cpp 

#include "udp_server.hpp"

#include <ctime>
#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>

#include "stdio.h"


using boost::asio::ip::udp;

std::string make_daytime_string()
{
  using namespace std; // For time_t, time and ctime;
  time_t now = time(0);
  return ctime(&now);
}


udp_server::udp_server(boost::asio::io_service& io_service, unsigned int port)
    : 
    m_socket(io_service, udp::endpoint(udp::v4(), port))
{
    start_receive();
}

void udp_server::start_receive()
{
    printf("udp_server::start_receive\n");
    m_socket.async_receive_from(
                                boost::asio::buffer(m_recv_buffer), 
                                m_remote_endpoint,
                                boost::bind(
                                            &udp_server::handle_receive, 
                                            this, 
                                            boost::asio::placeholders::error, 
                                            boost::asio::placeholders::bytes_transferred
                                           )
                               );
    printf("udp_server::start_receive DONE\n");
}


void udp_server::handle_receive(const boost::system::error_code& error, std::size_t nbytes)
{
    if (!error || error == boost::asio::error::message_size)
    {
        dump(nbytes);
 
        boost::shared_ptr<std::string> message(new std::string(make_daytime_string()));

        m_socket.async_send_to(
                               boost::asio::buffer(*message), 
                               m_remote_endpoint,
                               boost::bind(
                                           &udp_server::handle_send, 
                                           this, 
                                           message,
                                           boost::asio::placeholders::error,
                                           boost::asio::placeholders::bytes_transferred
                                          )
                              );
        start_receive();
    }
}

void udp_server::handle_send(
           boost::shared_ptr<std::string> message,
           const boost::system::error_code& error,
           std::size_t nbytes)
{
    printf("udp_server::handle_send %lu msg %s \n", nbytes, message->c_str());
}

void udp_server::dump(std::size_t nbytes )
{
    printf("udp_server::dump %lu\n", nbytes);
    for(unsigned int i=0;i<nbytes;i++) printf("%c",m_recv_buffer[i]);
    printf("\n");
}




