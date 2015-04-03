#pragma once

#include <string>
#include <iomanip>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include "udp_server.hpp"
#include "npy_server.hpp"

template <class Delegate>
class net_manager {

    boost::asio::io_service                          m_local_io_service ;
    boost::scoped_ptr<boost::asio::io_service::work> m_local_io_service_work ;
    boost::scoped_ptr<boost::thread>                 m_work_thread;
    udp_server<Delegate>                             m_udp_server ; 
    npy_server<Delegate>                             m_npy_server ; 

public:

   net_manager(Delegate* delegate, boost::asio::io_service& delegate_io_service, unsigned int udp_port, const char* zmq_backend)
       : 
       m_local_io_service(),
       m_local_io_service_work(new boost::asio::io_service::work(m_local_io_service)), 
       m_work_thread(new boost::thread(boost::bind(
                                &boost::asio::io_service::run, 
                                &m_local_io_service
                                   ))),
       m_udp_server(m_local_io_service, delegate, delegate_io_service, udp_port),
       m_npy_server(m_local_io_service, delegate, delegate_io_service, zmq_backend)
    {
#ifdef VERBOSE
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " net_manager::net_manager " << std::endl;
#endif
    }

};


