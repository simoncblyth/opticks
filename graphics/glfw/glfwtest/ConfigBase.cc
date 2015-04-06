#include "ConfigBase.hh"
#include <boost/tokenizer.hpp>

namespace po = boost::program_options;

#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <algorithm>



// A helper function to simplify the main part.
template<class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " ")); 
    return os;
}


ConfigBase::ConfigBase()
    :
    m_abort(false),
    m_config_file_default("demo.cfg"),
    m_liveline(""),
    m_generic("Generic Options only allowed on command line"),
    m_config("Configuration Options allowed both on command line and in config file"),
    m_hidden("Hidden Options allowed both on command line and in config file but not shown to user"),
    m_live("Live Options allowed via UDP message command line and config file"),
    m_cmdline_options("Command line options"),
    m_config_file_options("Config file options"),
    m_visible_options("Allowed options"),
    m_positional_options()
{
}


void ConfigBase::create_descriptions()
{
   m_generic.add_options()
       ("version,v", "print version string")
       ("help",      "produce help message")
       ("config,c",   
             po::value<std::string>(&m_config_file)->default_value(m_config_file_default),
             "name of a file of a configuration."
       )
       ("liveline",   
             po::value<std::string>(&m_liveline),
             "string with spaces to be live parsed, as test of composed overrides"
       )
       ;
 
}


bool ConfigBase::isAbort()
{
    return m_abort ;  
}


void ConfigBase::parse(int argc, char** argv)
{
    // store function will not change an already assigned value
    // thus for params that need to be live adjustable adopt
    // workaround of using composable and picking the last value
    //
    // hmm that means need different parsing orders for 
    // different variable sets ?
    //
    //  non-composable : commandline, configfile
    //  composable     : configfile,  commandline
    //
    //  

    std::cout << "ConfigBase::parse\n" ; 

    parse_commandline(argc, argv); 

    if(!m_config_file.empty())
    {
        parse_configfile(m_config_file.c_str(), m_config_file_options); 
    }

    if(!m_liveline.empty())
    {
        parse_liveline(m_liveline.c_str(), m_live); 
    }

    if (m_vm.count("help"))    std::cout << m_visible_options << "\n";
    if (m_vm.count("version")) std::cout << "version 1.0\n";
    if( m_vm.count("help") || m_vm.count("version")) m_abort = true ; 

    dump();
}


void ConfigBase::parse_commandline(int argc, char** argv)
{
    try {
         po::command_line_parser parser(argc, argv);
         parser.options(m_cmdline_options);
         parser.positional(m_positional_options);
         po::parsed_options opts = parser.run();
         dump(opts, "ConfigBase::parse_commandline");
         po::store(opts, m_vm);
         po::notify(m_vm);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << "\n";
    }    
}


void ConfigBase::parse_configfile(const char* path, boost::program_options::options_description& options)
{
    try {
        std::ifstream ifs(path);
        if (!ifs)
        {
            std::cout << "ConfigBase::parse_configfile failed to open: " << path << "\n";
        }
        else
        {
            po::parsed_options opts = po::parse_config_file(ifs, options);
            dump(opts, "ConfigBase::parse_configfile");
            po::store(opts, m_vm);
            po::notify(m_vm);
        }
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << "\n";
    }    
}


void ConfigBase::dump(po::parsed_options& opts, const char* msg )
{
    std::cout << msg << '\n' ;
    for(auto opt: opts.options)
    {
        std::cout << std::setw(20) << opt.string_key << " : " ;
        for(auto s: opt.value) std::cout << s << " " ;
        std::cout << '\n' ;
    }
}

void ConfigBase::parse_liveline(const char* _line)
{
    parse_liveline(_line, m_live);
}

void ConfigBase::parse_liveline(const char* _line, boost::program_options::options_description& desc)
{
   // hmm, there is a priority issue
   // as this gets done after commandline and config file have had their
   // chance and normally first setting wins

   std::string line(_line);
   //std::replace( line.begin(), line.end(), '_', ' ');   

   std::cout << "ConfigBase::parse_liveline [" << line << "]\n" ; 

   boost::char_separator<char> sep(" ");
   boost::tokenizer<boost::char_separator<char> > tok(line, sep);

  // typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;
  // boost::escaped_list_separator<char> sep( '\\', ' ', '\"' );
  // Tokenizer tok( line, sep );

   std::vector<std::string> args;
   std::copy(tok.begin(), tok.end(), std::back_inserter(args));
     
   for(size_t i=0 ; i < args.size() ; i++) std::cout << "ConfigBase::parse_liveline " << i << " " << args[i] << "\n" ; 

   po::command_line_parser parser(args);
   parser.options(desc);

   po::parsed_options opts = parser.run();

   dump(opts, "ConfigBase::parse_liveline");
   po::store(opts, m_vm);
   po::notify(m_vm);
}




void ConfigBase::dumpdesc(boost::program_options::options_description& desc, const char* msg)
{
    std::cout << "\nConfigBase::dumpdesc " << msg << std::endl ;
    for (auto opt: desc.options())
    {
        std::cout 
                << " format_name" << std::setw(30)     << opt->format_name() 
                << " format_paramter" << std::setw(30) << opt->format_parameter() 
                << std::endl;
    }    
}


void ConfigBase::dumpvm()
{

   /*
    dumpdesc(m_generic, "m_generic");
    dumpdesc(m_config,  "m_config");
    dumpdesc(m_hidden,  "m_hidden");
    dumpdesc(m_live,    "m_live");
    dumpdesc(m_cmdline_options,    "m_cmdline_options");
    dumpdesc(m_config_file_options, "m_config_file_options");
    dumpdesc(m_visible_options, "m_visible_options");
   */

    //dumpdesc(m_positional_options, "m_positional_options");


    for (po::variables_map::iterator it=m_vm.begin() ; it!=m_vm.end() ; it++)
    {
        std::string name = it->first ;
        po::variable_value var = it->second ;

        bool empty    = var.empty();
        bool defaulted = var.defaulted();

        std::cout << std::setw(20) << name 
                  << " empty " << empty 
                  << " defaulted " << defaulted 
                  << "\n"; 
    }
}






void ConfigBase::dump()
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

    dumpvm();

}
