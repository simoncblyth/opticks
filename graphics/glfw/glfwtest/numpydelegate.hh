#pragma once

#include <string>
#include <vector>
class Cfg;

template <class numpydelegate>
class numpyserver ;

class numpydelegate {
public:
   numpydelegate();
   void setServer(numpyserver<numpydelegate>* server);

   void on_msg(std::string addr, unsigned short port, std::string msg);
   void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata);

public:
   void addLiveCfg(Cfg* cfg);
   void interpretExternalMessage(std::string msg);

   void configureI(const char* name, std::vector<int>         values);
   void configureS(const char* name, std::vector<std::string> values);

   void setNPYEcho(int echo);
   int  getNPYEcho();

   void setUDPPort(int port);
   int  getUDPPort();

   void setZMQBackend(std::string& backend);
   std::string& getZMQBackend();

private:
    numpyserver<numpydelegate>* m_server ;    

   int         m_udp_port ;
   int         m_npy_echo ;
   std::string m_zmq_backend ;    

private:

   std::vector<Cfg*> m_live_cfg ; 


};


