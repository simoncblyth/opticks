#pragma once

#include <string>
#include <iomanip>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>


#include <asio-zmq.hpp>
#include "udp_server.hpp"
#include "npy_server.hpp"

/*
   net_manager : 
        no longer a good name for this, 
        its becoming communication hub of the app 

*/


template <class Delegate>
class net_manager : public boost::enable_shared_from_this<net_manager<Delegate>>  {

    boost::asio::zmq::context                        m_ctx;
    boost::asio::io_service                          m_local_io_service ;
    boost::scoped_ptr<boost::asio::io_service::work> m_local_io_service_work ;
    boost::scoped_ptr<boost::thread>                 m_work_thread;
    udp_server<Delegate>                             m_udp_server ; 
    npy_server<Delegate>                             m_npy_server ; 
    bool                                             m_npy_echo ;

public:

   net_manager(Delegate* delegate, boost::asio::io_service& delegate_io_service)
       : 
       m_local_io_service(),
       m_local_io_service_work(new boost::asio::io_service::work(m_local_io_service)), 
       m_work_thread(new boost::thread(boost::bind(
                                &boost::asio::io_service::run, 
                                &m_local_io_service
                                   ))),
       m_udp_server(m_local_io_service, delegate, delegate_io_service, delegate->getUDPPort()),
       m_npy_server(m_local_io_service, delegate, delegate_io_service, m_ctx ),
       m_npy_echo(delegate->getNPYEcho())
    {
#ifdef VERBOSE
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " net_manager::net_manager " << std::endl;
#endif
    }

    void send(std::string& addr, unsigned short port, std::string& msg)
    {

#ifdef VERBOSE
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " net_manager::send [" << msg << "] " 
                  << std::endl;
#endif
 
        // send by value to avoid grief 
        // huh, looks like args going out of scope but 
        // the bind is bundling them up by value ?

        std::string vaddr(addr);
        std::string vmsg(msg);
 
        m_local_io_service.post(
                   boost::bind(
                        &udp_server<Delegate>::send, 
                        &m_udp_server,
                        vaddr,
                        port,
                        vmsg 
                        ));
    }


    void response(std::vector<int>& shape,std::vector<float>& data,  std::string& metadata )
    {
        std::vector<int>   vshape(shape);
        std::vector<float> vdata(data);
        std::string        vmetadata(metadata);

        assert(!m_npy_echo); // response is invalid in npy_echo mode

        m_local_io_service.post(
                   boost::bind(
                        &npy_server<Delegate>::response, 
                        &m_npy_server,
                        vshape,
                        vdata,
                        vmetadata 
                        ));
    }





};


