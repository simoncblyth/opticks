// based on /usr/local/env/boost/basio/example/cpp03/tutorial/daytime6/server.cpp 

#include "udpServer.hh"

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


udpServer::udpServer(boost::asio::io_service& io_service, unsigned int port, EventQueue* queue, boost::mutex* iomutex)
    : 
    m_socket(io_service, udp::endpoint(udp::v4(), port)),
    m_queue(queue),
    m_iomutex(iomutex)
{
    start_receive();
}


void udpServer::start_receive()
{
    printf("udpServer::start_receive\n");
    m_socket.async_receive_from(
                                boost::asio::buffer(m_recv_buffer), 
                                m_remote_endpoint,
                                boost::bind(
                                            &udpServer::handle_receive, 
                                            this, 
                                            boost::asio::placeholders::error, 
                                            boost::asio::placeholders::bytes_transferred
                                           )
                               );
    printf("udpServer::start_receive DONE\n");
}


void udpServer::handle_receive(const boost::system::error_code& error, std::size_t nbytes)
{
    if (!error || error == boost::asio::error::message_size)
    {
        dump(nbytes);

        // locking push onto the queue as runs in separate netThread 
        // and need to avoid conflicts between writing to and 
        // reading from the queue

        {
            printf("handle_receive pushing onto queue\n");
            boost::mutex::scoped_lock lock(*m_iomutex);
            m_queue->push(m_recv_buffer);
        }
        m_recv_buffer.empty();
        //lock.unlock();

        if (error == boost::asio::error::message_size) {
           std::cout << "Message too large - Network Event";
        }   

        start_receive();
    }
}


void udpServer::send_time()
{
    boost::shared_ptr<std::string> message(new std::string(make_daytime_string()));
    m_socket.async_send_to(
                           boost::asio::buffer(*message), 
                           m_remote_endpoint,
                           boost::bind(
                                       &udpServer::handle_send, 
                                       this, 
                                       message,
                                       boost::asio::placeholders::error,
                                       boost::asio::placeholders::bytes_transferred
                                      )   
                          );  
}




void udpServer::handle_send(
           boost::shared_ptr<std::string> message,
           const boost::system::error_code& error,
           std::size_t nbytes)
{
    printf("udpServer::handle_send %lu msg %s \n", nbytes, message->c_str());
}

void udpServer::dump(std::size_t nbytes )
{
    printf("udpServer::dump %lu\n", nbytes);
    for(unsigned int i=0;i<nbytes;i++) printf("%c",m_recv_buffer[i]);
    printf("\n");
}




