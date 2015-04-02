#ifndef CONFIG_H
#define CONFIG_H

#include <boost/program_options.hpp>
#include <string>
#include <vector>

class Config {
    public:
        Config();
        void parse(int argc, char** argv);
        void parse(const char* line);
        void parse(std::vector<std::string>& lines, char delim);
        void split(std::vector<std::string>& elem, const char* line, char delim);

    public:
        int   getUdpPort();
        const char* getZMQBackend();
        void dump();

    public:
        float getYfov();
        bool  isAbort();

    private:
        void create_descriptions();

    private:
        boost::program_options::options_description m_generic ; 
        boost::program_options::options_description m_config  ; 
        boost::program_options::options_description m_hidden  ; 
        boost::program_options::options_description m_live  ; 
        boost::program_options::options_description m_cmdline_options  ; 
        boost::program_options::options_description m_config_file_options  ; 
        boost::program_options::options_description m_visible_options  ; 
        boost::program_options::positional_options_description m_positional_options ;
    private:
        boost::program_options::variables_map  m_vm;

    private:
        // hmm not very flexible, maybe maps of different types 
        std::string  m_config_file;
        std::string  m_config_file_default;
        std::string  m_zmq_backend;
        std::string  m_zmq_backend_default;

        int          m_opt ; 
        int          m_opt_default ; 
        int          m_udp_port ;
        int          m_udp_port_default ;
        float        m_yfov ;
        float        m_yfov_default ;
        bool         m_abort ;


};


#endif
