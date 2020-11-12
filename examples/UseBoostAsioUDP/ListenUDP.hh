#pragma once

#include <boost/asio.hpp>
class SCtrl ; 


class ListenUDP
{
    public:
        ListenUDP(boost::asio::io_context& io_context, SCtrl* ctrl_);
    private:
        void handle_receive_from(const boost::system::error_code& error,size_t bytes_recvd);
    private:
        SCtrl*      ctrl ; 
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


