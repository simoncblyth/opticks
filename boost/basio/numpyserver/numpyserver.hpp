#pragma once

#include "net_manager.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

template <class Delegate>
class numpyserver {

   boost::asio::io_service                              m_io_service ;
   boost::scoped_ptr<boost::asio::io_service::work>     m_io_service_work ;
   net_manager<Delegate>                                m_net_manager ; 

public:
   numpyserver(Delegate* delegate, unsigned int udp_port, const char* zmq_backend)  
      :
        m_io_service_work(new boost::asio::io_service::work(m_io_service)),
        m_net_manager(delegate, m_io_service, udp_port, zmq_backend)
   {

#ifdef VERBOSE
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " numpyserver::numpyserver " << std::endl;

#endif
   }

public:
   void sleep(unsigned int seconds);
   void poll_one();
   void poll();
   void stop();
   boost::asio::io_service& get_io_service();

};



template <typename Delegate>
void numpyserver<Delegate>::sleep(unsigned int seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

template <typename Delegate>
void numpyserver<Delegate>::poll_one()
{
    //std::cout << std::setw(20) << boost::this_thread::get_id() << " numpyserver::poll_one " << std::endl;
    m_io_service.poll_one();
} 

template <typename Delegate>
void numpyserver<Delegate>::poll()
{
    //std::cout << std::setw(20) << boost::this_thread::get_id() << " numpyserver::poll " << std::endl;
    m_io_service.poll();
} 

template <typename Delegate>
void numpyserver<Delegate>::stop()
{
    std::cout << std::setw(20) << boost::this_thread::get_id() << " numpyserver::stop " << std::endl;
    m_io_service_work.reset();
} 

template <typename Delegate>
boost::asio::io_service& numpyserver<Delegate>::get_io_service()
{
    return m_io_service ;  
}



