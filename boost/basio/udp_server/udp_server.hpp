#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#include <string>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>

#define BUFFER_SIZE 1024

class udp_manager ; 


class udp_server
{
public:
    udp_server(
           boost::asio::io_service& io_service, 
           boost::asio::io_service& parent_io_service, 
           udp_manager* udp_manager_,
           unsigned int port);

private:
    void start_receive();
    void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred );
    void handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code& /*error*/, std::size_t /*bytes_transferred*/);
    void dump(std::size_t nbytes );

private:
    boost::asio::ip::udp::socket    m_socket;
    boost::asio::ip::udp::endpoint  m_remote_endpoint;
    boost::array<char, BUFFER_SIZE> m_recv_buffer;
    boost::asio::io_service&        m_parent_io_service ;
    udp_manager*                    m_udp_manager ;
};



#endif
