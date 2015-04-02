#pragma once

#include "net_manager.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

template <class Delegate>
class net_server {

   boost::asio::io_service                              m_io_service ;
   boost::scoped_ptr<boost::asio::io_service::work>     m_io_service_work ;
   net_manager<Delegate>                                m_net_manager ; 

public:
   net_server(Delegate* delegate, unsigned int udp_port, const char* zmq_backend)  
      :
        m_io_service_work(new boost::asio::io_service::work(m_io_service)),
        m_net_manager(delegate, m_io_service, udp_port, zmq_backend)
   {
        std::cout << std::setw(20) << boost::this_thread::get_id() << " net_server::net_server " << std::endl;
   }

public:
   void sleep(unsigned int seconds);
   void poll_one();
   void stop();
   boost::asio::io_service& get_io_service();
   //virtual void on_msg(std::string msg);
   //virtual void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata);

};



template <typename Delegate>
void net_server<Delegate>::sleep(unsigned int seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

template <typename Delegate>
void net_server<Delegate>::poll_one()
{
    //std::cout << std::setw(20) << boost::this_thread::get_id() << " net_server::poll_one " << std::endl;
    m_io_service.poll_one();
} 

template <typename Delegate>
void net_server<Delegate>::stop()
{
    std::cout << std::setw(20) << boost::this_thread::get_id() << " net_server::stop " << std::endl;
    m_io_service_work.reset();
} 

template <typename Delegate>
boost::asio::io_service& net_server<Delegate>::get_io_service()
{
    return m_io_service ;  
}



