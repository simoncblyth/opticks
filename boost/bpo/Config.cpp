#include "Config.hh"
namespace po = boost::program_options;

#include <iostream>
#include <fstream>
#include <iterator>


void Config::parse(int argc, char** argv)
{
    try {
        store(po::command_line_parser(argc, argv)
                 .options(m_cmdline_options)
                 .positional(m_positional_options)
                 .run(), m_vm);
        notify(m_vm);
            
        std::ifstream ifs(m_config_file.c_str());
        if (!ifs)
        {
            std::cout << "can not open config file: " << m_config_file << "\n";
        }
        else
        {
            store(parse_config_file(ifs, m_config_file_options), m_vm);
            notify(m_vm);
        }

    }
    catch(std::exception& e)
    {
        std::cout << e.what() << "\n";
    }    
    act();
}

void Config::act()
{
    if (m_vm.count("help")) 
    {
        std::cout << m_visible_options << "\n";
        return ;
    }

    if (m_vm.count("version")) {
        std::cout << "Multiple sources example, version 1.0\n";
        return ;
    }

    dump();
}


// A helper function to simplify the main part.
template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " ")); 
    return os;
}



void Config::dump()
{
    
    if (m_vm.count("include-path"))
    {
        std::cout << "Include paths are: " 
                  << m_vm["include-path"].as< std::vector<std::string> >() << "\n";
    }

    if (m_vm.count("input-file"))
    {
        std::cout << "Input files are: " 
                  << m_vm["input-file"].as< std::vector<std::string> >() << "\n";
    }

    std::cout << "Optimization level is " << m_opt << "\n";                
}

Config::Config()
   : 
   m_generic("Generic Options only allowed on command line"),
   m_config("Configuration Options allowed both on command line and in config file"),
   m_hidden("Hidden Options allowed both on command line and in config file but not shown to user"),
   m_cmdline_options("Command line options"),
   m_config_file_options("Config file options"),
   m_visible_options("Allowed options"),
   m_positional_options(),
   m_config_file_default("multiple_sources.cfg"),
   m_opt_default(10)
{
   create_descriptions();
}

void Config::create_descriptions()
{
   m_generic.add_options()
       ("version,v", "print version string")
       ("help",      "produce help message")
       ("config,c",   
             po::value<std::string>(&m_config_file)->default_value(m_config_file_default),
             "name of a file of a configuration."
       )
       ;
    
   m_config.add_options()
       ("optimization",   
               po::value<int>(&m_opt)->default_value(m_opt_default), 
               "optimization level"
       )
       ("include-path,I", 
               po::value< std::vector<std::string> >()->composing(), 
               "include path"
       )
       ;

   m_hidden.add_options()
       ("input-file", 
               po::value< std::vector<std::string> >(),  
               "input file"
       )
       ;

   m_cmdline_options.add(m_generic).add(m_config).add(m_hidden);
   m_config_file_options.add(m_config).add(m_hidden);
   m_visible_options.add(m_generic).add(m_config);
   m_positional_options.add("input-file", -1);
}


int main(int argc, char** argv)
{
     Config config;
     config.parse(argc,argv);

     return 0 ;
}

