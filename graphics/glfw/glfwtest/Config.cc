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


    if (m_vm.count("help")) 
    {
        std::cout << m_visible_options << "\n";
        m_abort = true ; 
    }

    if (m_vm.count("version")) {
        std::cout << "version 1.0\n";
        m_abort = true ; 
    }

    dump();
}


void Config::parse(const char* line)
{
    std::vector<std::string> elem ;
    char delim = ' ' ;

    std::istringstream iss(line);
    std::string s;
    while (getline(iss, s, delim))
    {
        elem.push_back(s);
    }
}

void Config::split(std::vector<std::string>& elem, const char* line, char delim)
{
    std::istringstream iss(line);
    std::string s;
    while (getline(iss, s, delim))
    {
        elem.push_back(s);
    }
}


void Config::parse(std::vector<std::string>& lines, char delim)
{
    std::vector<std::string> tokens ;
    for(unsigned int i=0 ; i<lines.size() ; ++i)
    {
        split(tokens, lines[i].c_str(), delim);

    }
    
    for(unsigned int i=0 ; i<tokens.size() ; ++i)
    {
        printf("Config::parse %d %s \n", i, tokens[i].c_str() );
    }
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

    // hmm need a structured way of doing this, 
    // holding values together with defaults 
    std::cout << "Optimization level is " << m_opt << "\n";                
    std::cout << "UDP port level is " << m_udp_port << "\n";                
    std::cout << "Yfov               " << m_yfov << "\n";                
}

Config::Config()
    : 
    m_generic("Generic Options only allowed on command line"),
    m_config("Configuration Options allowed both on command line and in config file"),
    m_hidden("Hidden Options allowed both on command line and in config file but not shown to user"),
    m_live("Live Options allowed via UDP message command line and config file"),
    m_cmdline_options("Command line options"),
    m_config_file_options("Config file options"),
    m_visible_options("Allowed options"),
    m_positional_options(),
    m_config_file_default("multiple_sources.cfg"),
    m_opt_default(10),
    m_udp_port(8080),
    m_udp_port_default(8080),
    m_yfov(60.f),
    m_yfov_default(60.f),
    m_abort(false)
{
    create_descriptions();
}


int Config::getUdpPort()
{
    return m_udp_port ;
}

float Config::getYfov()
{
    return m_yfov ;  
}

bool Config::isAbort()
{
    return m_abort ;  
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
       ("udp-port",   
               po::value<int>(&m_udp_port)->default_value(m_udp_port_default), 
               "UDP port on which to listen for external messages"
       )
       ("include-path,I", 
               po::value< std::vector<std::string> >()->composing(), 
               "include path"
       )
       ;


   m_live.add_options()
       ("yfov",   
               po::value<float>(&m_yfov)->default_value(m_yfov_default), 
               "Vertical Field of view in degrees"
       )
       ;

   m_hidden.add_options()
       ("input-file", 
               po::value< std::vector<std::string> >(),  
               "input file"
       )
       ;

   m_cmdline_options.add(m_generic).add(m_config).add(m_hidden).add(m_live);
   m_config_file_options.add(m_config).add(m_hidden).add(m_live);
   m_visible_options.add(m_generic).add(m_config).add(m_live);
   m_positional_options.add("input-file", -1);
}



