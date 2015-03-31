#ifndef UDPSERVER_H
#define UDPSERVER_H

#include <string>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include "EventQueue.hh"

class udpServer
{
public:
    udpServer(boost::asio::io_service& io_service, unsigned int port, EventQueue* queue, boost::mutex* iomutex);

private:
    void send_time();
    void start_receive();
    void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred );
    void handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code& /*error*/, std::size_t /*bytes_transferred*/);
    void dump(std::size_t nbytes );

private:
    boost::asio::ip::udp::socket    m_socket;
    boost::asio::ip::udp::endpoint  m_remote_endpoint;
    boost::mutex*                   m_iomutex;
private:
    EventQueueItem_t                m_recv_buffer;
    EventQueue*                     m_queue ; 
};


#endif
