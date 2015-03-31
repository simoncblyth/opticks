#ifndef CONFIG_H
#define CONFIG_H

#include <boost/program_options.hpp>
#include <string>


class Config {
    public:
        Config();
        void parse(int argc, char** argv);

    private:
        void create_descriptions();
        void act();
        void dump();

    private:
        boost::program_options::options_description m_generic ; 
        boost::program_options::options_description m_config  ; 
        boost::program_options::options_description m_hidden  ; 
        boost::program_options::options_description m_cmdline_options  ; 
        boost::program_options::options_description m_config_file_options  ; 
        boost::program_options::options_description m_visible_options  ; 
        boost::program_options::positional_options_description m_positional_options ;
        boost::program_options::variables_map  m_vm;

    private:
        std::string  m_config_file;
        std::string  m_config_file_default;
        int          m_opt ; 
        int          m_opt_default ; 

};


#endif
