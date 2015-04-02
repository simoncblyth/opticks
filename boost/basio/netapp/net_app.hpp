#pragma once

#include "net_manager.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

class net_app {

   boost::asio::io_service                              m_io_service ;
   boost::scoped_ptr<boost::asio::io_service::work>     m_io_service_work ;
   net_manager<net_app>                                 m_net_manager ; 

public:
   net_app(unsigned int udp_port, const char* zmq_backend)  
      :
        m_io_service_work(new boost::asio::io_service::work(m_io_service)),
        m_net_manager(this, udp_port, zmq_backend)
   {
        std::cout << std::setw(20) << boost::this_thread::get_id() << " net_app::net_app " << std::endl;
   }

public:
   void sleep(unsigned int seconds);
   void poll_one();
   void stop();
   boost::asio::io_service& get_io_service();
   virtual void on_msg(std::string msg);
   virtual void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata);

};



void net_app::sleep(unsigned int seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

void net_app::poll_one()
{
    std::cout << std::setw(20) << boost::this_thread::get_id() << " net_app::poll_one " << std::endl;
    m_io_service.poll_one();
} 

void net_app::stop()
{
    std::cout << std::setw(20) << boost::this_thread::get_id() << " net_app::stop " << std::endl;
    m_io_service_work.reset();
} 

void net_app::on_msg(std::string msg)
{
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " net_app::on_msg " 
              << " msg ["  << msg << "] "
              << std::endl;
}

void net_app::on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata)
{
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " net_app::on_npy " 
              << " shape dimension " << shape.size()
              << " data size "  << data.size()
              << " metadata [" << metadata << "]" 
              << std::endl;
}

boost::asio::io_service& net_app::get_io_service()
{
    return m_io_service ;  
}


