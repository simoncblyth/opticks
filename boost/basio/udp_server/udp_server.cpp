// based on /usr/local/env/boost/basio/example/cpp03/tutorial/daytime6/server.cpp 

#include "udp_server.hpp"
#include "udp_manager.hpp"

#include <ctime>
#include <iostream>
#include <string>
#include <iomanip>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include "stdio.h"


using boost::asio::ip::udp;

std::string make_daytime_string()
{
  using namespace std; // For time_t, time and ctime;
  time_t now = time(0);
  return ctime(&now);
}


udp_server::udp_server(
             boost::asio::io_service& io_service, 
             boost::asio::io_service& parent_io_service, 
             udp_manager*             udp_manager_,
             unsigned int port
           )
    : 
    m_socket(io_service, udp::endpoint(udp::v4(), port)),
    m_parent_io_service(parent_io_service),
    m_udp_manager(udp_manager_)
{
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::udp_server " << std::endl;

    start_receive();
}

void udp_server::start_receive()
{
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::start_receive " << std::endl;
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

    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::start_receive DONE " << std::endl;
}


void udp_server::handle_receive(const boost::system::error_code& error, std::size_t nbytes)
{

//    std::cout 
//         << std::setw(20) << boost::this_thread::get_id() 
//         << " udp_server::handle_receive " << std::endl;


    if (!error || error == boost::asio::error::message_size)
    {
        dump(nbytes);


        // surely just leaking should always work
        std::string* smsg = new std::string(m_recv_buffer.begin(), m_recv_buffer.begin() + nbytes);
        boost::shared_ptr<std::string> msg(smsg);
        std::cout 
             << std::setw(20) << boost::this_thread::get_id() 
             << " udp_server::handle_receive " 
             << " smsg [" << *smsg << "] "
             << " msg [" << *msg << "] "
             << std::endl;

/*
        m_parent_io_service.post(
                        boost::bind(
                                &udp_manager::on_message_0,
                                m_udp_manager,
                                msg ));
*/

        m_parent_io_service.post(
                        boost::bind(
                                &udp_manager::on_message,
                                m_udp_manager,
                                *smsg ));


 
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
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::handle_send " 
         << " nbytes " << nbytes
         << " msg " << message
         << std::endl;
}

void udp_server::dump(std::size_t nbytes )
{
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::dump " 
         << " nbytes " << nbytes
         << std::endl;

    printf("udp_server::dump %lu\n", nbytes);
    for(unsigned int i=0;i<nbytes;i++) printf("%c",m_recv_buffer[i]);
    printf("\n");
}




