/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <string>
#include <vector>
class Cfg;
class NumpyEvt;

template <class numpydelegate>
class numpyserver ;


class numpydelegate {
public:
   numpydelegate();
   void setServer(numpyserver<numpydelegate>* server);
   void setNumpyEvt(NumpyEvt* evt);

   void on_msg(std::string addr, unsigned short port, std::string msg);
   void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata);

public:
   void liveConnect(Cfg* cfg);
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
    NumpyEvt*         m_evt ; 


};


