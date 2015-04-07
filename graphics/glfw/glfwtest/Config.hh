#ifndef CONFIG_H
#define CONFIG_H

#include "ConfigBase.hh"

// split into application specific and general config handling in ConfigBase



/*
class CameraCfg {
  public:
        CameraCfg();
        float       getYfov();

        void Yfov_set(std::vector<float> yfov);

  private:
        float        m_yfov ;
        float        m_yfov_default ;


};
*/



class Config : public ConfigBase  {
    public:
        Config();

    public:
        void        dump();
        int         getUdpPort();
        const char* getZMQBackend();
        float       getYfov();

        void Yfov_set(std::vector<float> yfov);




    private:
        void create_descriptions();

    private:
        // hmm not very flexible, maybe maps of different types 
        std::string  m_zmq_backend;
        std::string  m_zmq_backend_default;

        int          m_udp_port ;
        int          m_udp_port_default ;

        float        m_yfov ;
        float        m_yfov_default ;

};


#endif
