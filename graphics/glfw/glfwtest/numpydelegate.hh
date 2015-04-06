#pragma once

#include <string>
#include <vector>


template <class numpydelegate>
class numpyserver ;

class numpydelegate {
public:
   numpydelegate();
   void setServer(numpyserver<numpydelegate>* server);

   void on_msg(std::string addr, unsigned short port, std::string msg);
   void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata);

private:
    numpyserver<numpydelegate>* m_server ;    



};


