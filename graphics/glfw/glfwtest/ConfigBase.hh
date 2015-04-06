#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>

class ConfigBase {

        boost::program_options::variables_map  m_vm;
        bool                                   m_abort ;

   protected:
        std::string  m_config_file;
        std::string  m_config_file_default;
        std::string  m_liveline;

   protected:
        boost::program_options::options_description m_generic ; 
        boost::program_options::options_description m_config  ; 
        boost::program_options::options_description m_hidden  ; 
        boost::program_options::options_description m_live  ; 

        boost::program_options::options_description m_cmdline_options  ; 
        boost::program_options::options_description m_config_file_options  ; 
        boost::program_options::options_description m_visible_options  ; 
        boost::program_options::positional_options_description m_positional_options ;

   public:
        ConfigBase();
        void create_descriptions();
        bool isAbort();

   public:
        void parse(int argc, char** argv);
        void parse_commandline(int argc, char** argv);
        void parse_configfile(const char* path, boost::program_options::options_description& options);
        void parse_liveline(const char* _line);
        void parse_liveline(const char* _line, boost::program_options::options_description& options);


        virtual void dumpvm();
        virtual void dump();
        void dumpdesc(boost::program_options::options_description& desc, const char* msg);
        void dump(boost::program_options::parsed_options& opts, const char* msg );


};



