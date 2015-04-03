#pragma once
// started from /usr/local/env/boost/basio/example/cpp03/tutorial/daytime6/server.cpp 

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

//#define VERBOSE 1


template <class Delegate> 
class udp_server
{
    boost::asio::ip::udp::socket    m_socket;
    boost::asio::ip::udp::endpoint  m_remote_endpoint;
    boost::array<char,1024>         m_recv_buffer;
    Delegate*                       m_delegate  ;
    boost::asio::io_service&        m_delegate_io_service ;

public:
    udp_server(
           boost::asio::io_service& io_service, 
           Delegate*                delegate,
           boost::asio::io_service& delegate_io_service, 
           unsigned int port
           )
    : 
       m_socket(io_service, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port)),
       m_delegate(delegate),
       m_delegate_io_service(delegate_io_service)

    {
#ifdef VERBOSE
        std::cout 
             << std::setw(20) << boost::this_thread::get_id() 
             << " udp_server::udp_server " << std::endl;
#endif
        start_receive();
    } 

private:
    void start_receive();
    void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred );
    void handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code& /*error*/, std::size_t /*bytes_transferred*/);
    void dump(std::size_t nbytes );

};




template <typename Delegate>
void udp_server<Delegate>::start_receive()
{
#ifdef VERBOSE
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::start_receive " << std::endl;
#endif
    m_socket.async_receive_from(
                            boost::asio::buffer(m_recv_buffer), 
                            m_remote_endpoint,
                            boost::bind(
                                        &udp_server<Delegate>::handle_receive, 
                                        this, 
                                        boost::asio::placeholders::error, 
                                        boost::asio::placeholders::bytes_transferred
                                       )
                           );

#ifdef VERBOSE
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::start_receive DONE " << std::endl;
#endif
}  



template <typename Delegate>
void udp_server<Delegate>::handle_receive(const boost::system::error_code& error, std::size_t nbytes)
{
    if (!error || error == boost::asio::error::message_size)
    {
        dump(nbytes);

        // surely just leaking should always work
        std::string* smsg = new std::string(m_recv_buffer.begin(), m_recv_buffer.begin() + nbytes);
        boost::shared_ptr<std::string> msg(smsg);
#ifdef VERBOSE
        std::cout 
             << std::setw(20) << boost::this_thread::get_id() 
             << " udp_server::handle_receive " 
             << " smsg [" << *smsg << "] "
             << " msg [" << *msg << "] "
             << std::endl;
#endif

        // msg back to the delegate, typically passing from work thread to main thread
        m_delegate_io_service.post(
                        boost::bind(
                                &Delegate::on_msg,
                                m_delegate,
                                *smsg ));


        std::time_t now = std::time(0);
        boost::shared_ptr<std::string> message(new std::string(std::ctime(&now)));

        m_socket.async_send_to(
                               boost::asio::buffer(*message), 
                               m_remote_endpoint,
                               boost::bind(
                                           &udp_server<Delegate>::handle_send, 
                                           this, 
                                           message,
                                           boost::asio::placeholders::error,
                                           boost::asio::placeholders::bytes_transferred
                                          )
                              );
        start_receive();
    }
}

template <typename Delegate>
void udp_server<Delegate>::handle_send(
           boost::shared_ptr<std::string> message,
           const boost::system::error_code& error,
           std::size_t nbytes)
{
#ifdef VERBOSE
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::handle_send " 
         << " nbytes " << nbytes
         << " msg " << message
         << std::endl;
#endif
}

template <typename Delegate>
void udp_server<Delegate>::dump(std::size_t nbytes )
{
#ifdef VERBOSE
    std::cout 
         << std::setw(20) << boost::this_thread::get_id() 
         << " udp_server::dump " 
         << " nbytes " << nbytes
         << std::endl;

    printf("udp_server::dump %lu\n", nbytes);
    for(unsigned int i=0;i<nbytes;i++) printf("%c",m_recv_buffer[i]);
    printf("\n");
#endif

}



