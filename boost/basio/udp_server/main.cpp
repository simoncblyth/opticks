/*
Test with eg::

        udpserver-test 
        UDP_PORT=13 udp.py hello


        npysend.sh --tag 1  # requires zmq- ; zmq-broker 

*/

#include <iostream>
#include "net_app.hpp"

class App : public net_app {
public:
   App() :
        net_app(13, "tcp://127.0.0.1:5002")
   { 
   }
   void on_msg(std::string msg)
   {
       std::cout 
           << "App::on_msg "
           << "msg [" << msg << "]" 
           << std::endl ; 
   }
   void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata)
   {
        std::cout 
           << "App::on_npy "
           << "shape dimension " << shape.size()
           << "data size " << data.size()
           << "metadata [" << metadata << "]" 
           << std::endl ; 
   } 

};


int main()
{
    App app ;
    for(unsigned int i=0 ; i < 20 ; ++i )
    {
        app.poll_one();
        app.sleep(1);
    }
    app.stop();

    return 0;
}

