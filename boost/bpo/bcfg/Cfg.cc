#include "Cfg.hh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>

typedef std::vector<std::string>::const_iterator VSI ; 

#include <boost/tokenizer.hpp>

namespace po = boost::program_options;



Cfg::Cfg(const char* name, bool live)
    : 
    m_desc(name), 
    m_name(strdup(name)),
    m_live(live),
    m_error(false),
    m_verbose(false)
{
}



bool Cfg::containsOthers()
{
    return !m_others.empty();
}

unsigned int Cfg::getNumOthers()
{
    return m_others.size();
}

Cfg* Cfg::getOther(unsigned int index)
{
    return m_others[index];
}

Cfg* Cfg::operator [](const char* name)
{
   Cfg* other = findOther(name);
   return other ; 
}

Cfg* Cfg::findOther(const char* name)
{
    for(size_t i=0 ; i < m_others.size() ; i++)
    {
        Cfg* other = m_others[i];
        if(strcmp(name, other->getName())==0) return other ;
    }
    return NULL;
}

void Cfg::add(Cfg* other)
{
    m_desc.add(other->getDesc());
    m_others.push_back(other);
}

boost::program_options::options_description& Cfg::getDesc()
{
    return m_desc ; 
}

std::string Cfg::getDescString()
{
    std::stringstream ss ; 
    ss << getDesc() ;
    return ss.str();
}


void Cfg::dumpTree(const char* msg)
{
    printf("Cfg::dumpTree %s \n", msg);
    dumpTree_(0);
}

void Cfg::dumpTree_(unsigned int depth)
{
    unsigned int nchild = m_others.size() ;
    printf("Cfg::dumpTree_ depth %d name %30s nchild %u \n", depth, m_name, nchild );  
    for(size_t i=0 ; i < nchild ; i++)
    {
        Cfg* other = m_others[i];
        other->dumpTree_(depth+1);
    } 
}





void Cfg::commandline(int argc, char** argv)
{
    if(m_verbose)
    std::cout << "Cfg::commandline " << m_name << std::endl ; 

    std::stringstream ss ; 
    for(int i=1 ; i < argc ; ++i ) ss << argv[i] << " " ;
    m_commandline = ss.str();

    if(m_others.size() == 0)
    {
        std::vector<std::string> unrecognized = parse_commandline(argc, argv);
        if(m_verbose) dump(unrecognized, "unrecognized after parse_commandline"); 
    }
    else
    {
        for(size_t i=0 ; i < m_others.size() ; i++)
        {
            Cfg* other = m_others[i];
            other->commandline(argc, argv);
        } 
    }
}

void Cfg::liveline(const char* _line)
{
    if(m_others.empty())
    {
        std::vector<std::string> unrecognized = parse_liveline(_line);

        if(m_verbose)
        {
            printf("Cfg::liveline %s \n", _line);
            dump(unrecognized, "Cfg::liveline unrecognized "); 
        }
    }
    else
    {
        for(size_t i=0 ; i < m_others.size() ; i++)
        {
            Cfg* other = m_others[i];
            if(other->isLive())
            {
                other->liveline(_line);
            }
        } 
    }

}
void Cfg::configfile(const char* path)
{
    if(m_others.empty())
    {
        std::vector<std::string> unrecognized = parse_configfile(path);
        if(m_verbose)
        dump(unrecognized, "unrecognized after parse_configfile"); 
    }
    else
    {
        for(size_t i=0 ; i < m_others.size() ; i++)
        {
            Cfg* other = m_others[i];
            other->configfile(path);
        } 
    }
}



std::vector<std::string> Cfg::parse_liveline(const char* _line)
{
   std::string line(_line);

   if(m_verbose)
   std::cout << "Cfg::parse_liveline [" << line << "]\n" ; 

   boost::char_separator<char> sep(" ");
   boost::tokenizer<boost::char_separator<char> > tok(line, sep);

   std::vector<std::string> tokens;
   std::copy(tok.begin(), tok.end(), std::back_inserter(tokens));
     
   return parse_tokens(tokens);
}


std::vector<std::string> Cfg::parse_commandline(int argc, char** argv, bool verbose)
{
    if(m_verbose || verbose) std::cout << "Cfg::parse_commandline " << m_name << std::endl ;  

    std::vector<std::string> unrecognized ; 
    try  
    {
         po::command_line_parser parser(argc, argv);
         parser.options(m_desc);
         parser.allow_unregistered();
         po::parsed_options parsed = parser.run();

         po::store(parsed, m_vm);
         po::notify(m_vm);

         std::vector<std::string> unrec = po::collect_unrecognized(parsed.options, po::include_positional); 

         unrecognized.assign(unrec.begin(), unrec.end());
    }
    catch(std::exception& e)
    {
        m_error = true ; 
        m_error_message = e.what();
        std::cout << e.what() << "\n";
    }    
  
    if(m_verbose || verbose) 
              std::cout << "Cfg::parse_commandline " << m_name << " DONE " 
              << " error " << m_error
              << " error_message " << m_error_message
              << std::endl ;

    

    if(m_verbose || verbose)
    {
        std::cout << "Cfg::parse_commandline unrecognized by " << m_name << ": ";  
        for(VSI it=unrecognized.begin() ; it != unrecognized.end() ; it++ ) std::cout << " " << *it ; 
        std::cout << std::endl ; 
    }
  

    return unrecognized ; 
}



std::vector<std::string> Cfg::parse_tokens(std::vector<std::string>& tokens)
{
    std::vector<std::string> unrecognized ; 
#ifdef VERBOSE
    dump(tokens, "Cfg::parse_tokens input");
#endif

    po::command_line_parser parser(tokens);
    parser.options(m_desc);
    parser.allow_unregistered();
    po::parsed_options parsed = parser.run();

#ifdef VERBOSE
    dump(parsed, "Cfg::parse_tokens parsed");
#endif
    po::store(parsed, m_vm);
    po::notify(m_vm);

    std::vector<std::string> unrec = po::collect_unrecognized(parsed.options, po::include_positional); 
    unrecognized.assign(unrec.begin(), unrec.end());

    return unrecognized ;
}





std::vector<std::string> Cfg::parse_configfile(const char* path)
{
    std::vector<std::string> unrecognized ; 
    try {
        std::ifstream ifs(path);
        if (!ifs)
        {
            std::cout << "Cfg::parse_configfile failed to open: " << path << "\n";
        }
        else
        {
            bool allow_unregistered = true ; 
            po::parsed_options parsed = po::parse_config_file(ifs, m_desc, allow_unregistered);
#ifdef VERBOSE
            dump(parsed, "Cfg::parse_configfile");
#endif
            po::store(parsed, m_vm);
            po::notify(m_vm);


#ifdef VERBOSE
            for (const auto& opt : parsed.options) {
                if (m_vm.find(opt.string_key) == m_vm.end()) {
                    std::cout << "Cfg::parse_configfile unrecognized option " << opt.string_key  << '\n' ;
                }
            }
#endif
        }
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << "\n";
    }    
    return unrecognized ;
}



void Cfg::dump(const char* msg)
{
    dump(m_desc, msg);
    dump(m_vm, msg);
}

void Cfg::dump(std::vector<std::string>& ss, const char* msg )
{
   std::cout << msg << " vec with " << ss.size() << " strings : " ; 
   for(VSI it=ss.begin() ; it != ss.end() ; it++) std::cout << "[" << *it << "]" ; 
   std::cout << std::endl ; 
}

void Cfg::dump(boost::program_options::parsed_options& opts, const char* msg )
{
    std::cout << msg << '\n' ;

    std::vector<po::basic_option<char>> options = opts.options ;  
    for(unsigned int i=0 ; i < options.size() ; ++i )
    {
        po::basic_option<char>& opt = options[i] ; 
        std::cout << std::setw(20) << "( " << opt.string_key << " : " ;
#ifdef VERBOSE
        for(auto s: opt.value) std::cout << s << " " ;
#endif
        std::cout << ") " << std::endl  ;
    }
}

void Cfg::dump(boost::program_options::options_description& desc, const char* msg)
{
    std::cout << "\nCfg::dumpdesc " << msg << std::endl ;
    // for (auto opt: desc.options())
    // {
    //     std::cout 
    //             << " format_name" << std::setw(30)     << opt->format_name() 
    //             << " format_paramter" << std::setw(30) << opt->format_parameter() 
    //             << std::endl;
    // }    
}

void Cfg::dump(boost::program_options::variables_map& vm, const char* msg)
{
    std::cout << msg << std::endl ;
    for (po::variables_map::iterator it=vm.begin() ; it!=vm.end() ; it++)
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



