#include <sstream>
#include <boost/algorithm/string.hpp>

// brap-
#include "BMap.hh"
#include "BFile.hh"

// npy-
#include "NState.hpp"
#include "NConfigurable.hpp"


#include "PLOG.hh"



NState::NState(const char* dir, const char* name) 
    :
    m_verbose(false),
    m_dir(strdup(dir)),
    m_name(strdup(name)),
    m_num_changes(0)
{
    init();
}


void NState::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}

void NState::setName(const char* name)
{
    free((void*)m_name);
    m_name = strdup(name);
}

const char* NState::getName()
{
   return m_name ; 
}
const char* NState::getDir()
{
   return m_dir ; 
}


void NState::setNumChanges(unsigned int num_changes)
{
    m_num_changes = num_changes ; 
}
unsigned int NState::getNumChanges()
{
    return m_num_changes ; 
}




NState* NState::load(const char* dir, unsigned int num)
{
    std::string name = FormName(num) ;
    NState* state = new NState(dir, name.c_str());
    state->load();
    return state ; 
}


void NState::init()
{
    std::string dir = BFile::FormPath(m_dir);
    if(strcmp(dir.c_str(),m_dir)!=0)
    {
        free((void*)m_dir);
        m_dir = strdup(dir.c_str()); 
        LOG(debug) << "NState::init promoted m_dir to " << m_dir ; 
    }
}


std::string NState::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " << m_dir << " " << m_name ; 
    return ss.str();
}  


std::string NState::FormName(unsigned int num)
{
    std::stringstream ss ; 
    ss << std::setfill('0') << std::setw(3) << num ;
    return ss.str() ;     
}

void NState::setName(unsigned int num)
{
    std::string name = FormName(num);
    setName(name.c_str());    
}

void NState::Summary(const char* msg)
{
    LOG(info) << description(msg);
}


void NState::addConfigurable(NConfigurable* configurable)
{
    const char* prefix = configurable->getPrefix();
    m_configurables[prefix] = configurable ; 
}

NConfigurable* NState::getConfigurable(const char* prefix)
{
    return m_configurables.count(prefix) == 1 ? m_configurables[prefix] : NULL ;
}

std::string NState::get(const char* key)
{
    std::string empty ;
    return m_kv.count(key) > 0 ? m_kv[key] : empty ; 
}

void NState::set(const char* key, const char* val)
{
    m_kv[key] = val ; 
}


std::string NState::formKey(const char* prefix, const char* tag)
{
    char key[64];
    snprintf(key, 64, "%s.%s", prefix, tag );
    return key ;  
}

unsigned int NState::splitKey(std::vector<std::string>& prefix_tag, const char* key)
{
    boost::split(prefix_tag, key, boost::is_any_of(".")); 
    //assert(prefix_tag.size() == 2 && "NState::splitKey malformed key, expecting form \"prefix.tag\"" ); 
    return prefix_tag.size() ; 
}


void NState::collect()
{
    if(m_verbose) LOG(info) << "NState::collect" ; 

    unsigned int changes(0);
    typedef std::map<std::string, NConfigurable*>::const_iterator SCI ; 
    for(SCI it=m_configurables.begin() ; it != m_configurables.end() ; it++)
    {
       changes += collect(it->second) ;
    }
    setNumChanges(changes) ; 
}

unsigned int NState::collect(NConfigurable* configurable)
{
    // NConfigurable is an abstract get/set/getTags/accepts/configure protocol 
    if(!configurable) return 0 ; 

    std::string empty ;
    std::vector<std::string> tags = configurable->getTags();

    const char* prefix = configurable->getPrefix();
    
    if(m_verbose)
    LOG(info) << "NState::collect" 
              << " prefix " << prefix 
              ; 

    unsigned int changes(0);

    for(unsigned int i=0 ; i < tags.size(); i++)
    {
        const char* tag = tags[i].c_str();

        std::string key = formKey(prefix, tag);
        std::string val = configurable->get(tag);
        std::string prior = get(key.c_str());
       
        if(m_verbose)  
        LOG(info) << "NState::collect"
                  << " key " <<  key.c_str()
                  << " val " << val.c_str()
                  << " prior " << prior.c_str()
                  ;

        if(prior.empty())
        {
            changes += 1 ; 
            set( key.c_str(), val.c_str() );
        }
        else if (strcmp(prior.c_str(), val.c_str())==0)
        {
            LOG(debug) << "NState::collect"
                      << " unchanged " 
                      << " key " <<  key.c_str()
                      << " val " << val.c_str()
                      ;
                   
        }
        else
        {
            changes += 1 ; 
            set( key.c_str(), val.c_str() );
        }

    }
    return changes ; 
}


void NState::apply()
{
    if(m_verbose)
    LOG(info) << "NState::apply " ;  

    typedef std::map<std::string, std::string>::const_iterator SSI ; 
    for(SSI it=m_kv.begin() ; it != m_kv.end() ; it++)
    {
         std::string key = it->first ;           
         std::string val = it->second ;

         std::vector<std::string> prefix_tag ; 
         unsigned int n = splitKey(prefix_tag, key.c_str());

         if(m_verbose)
         LOG(info) << "NState::apply " << key << "," << val << " n=" << n ;  

         if(n == 2)
         {
             std::string prefix = prefix_tag[0] ; 
             std::string tag = prefix_tag[1] ; 

             NConfigurable* configurable = getConfigurable(prefix.c_str());
             if(configurable)
             {

                std::string before = configurable->get(tag.c_str());
                if(strcmp(before.c_str(), val.c_str())!=0)
                {
                    if(m_verbose)
                    LOG(info) << "NState::apply " 
                              << " [" << getName() << "] "
                              << key 
                              << " change " 
                              << before  
                              << " --> " 
                              << val 
                              ;

                    configurable->configure(tag.c_str(), val.c_str());
                } 
             }
             else
             {
                if(m_verbose)
                LOG(warning) << "NState::apply no configurable for prefix " << prefix ;  
             }
         }
         else
         {
             LOG(warning) << "NState::apply skipped key " << key ;  
         }


    }
}


std::string NState::stateString()
{
    std::stringstream ss ; 
    typedef std::map<std::string, std::string>::const_iterator SSI ; 
    for(SSI it=m_kv.begin() ; it != m_kv.end() ; it++)
    {
         std::string key = it->first ;           
         std::string val = it->second ;

         std::vector<std::string> prefix_tag ; 
         unsigned int n = splitKey(prefix_tag, key.c_str());
         if(n!=2) continue ; 
         
         std::string prefix = prefix_tag[0] ; 
         std::string tag = prefix_tag[1] ; 

         ss << std::setw(20) << prefix 
            << std::setw(20) << tag 
            << std::setw(20) << val 
            << std::endl ; 
    }
    return ss.str();
}


const std::string& NState::getStateString(bool update_)
{
    if(m_state_string.empty() || update_)
    {
        m_state_string = stateString();
    }
    return m_state_string ; 
}

void NState::update()
{
    collect();
    getStateString(true);
}

std::string NState::getFileName()
{
   std::stringstream ss ; 
   ss << m_name << ".ini" ; 
   return ss.str(); 
}

void NState::save()
{
    update();

    std::string filename = getFileName();
    std::string path = BFile::preparePath(m_dir, filename.c_str());
    LOG(debug) << "NState::save " << path ;  
    BMap<std::string, std::string>::save(&m_kv, m_dir, filename.c_str() ) ;
}

int NState::load()
{
    std::string filename = getFileName();
    unsigned int depth = 1 ; 
    std::string path = BFile::preparePath(m_dir, filename.c_str());
    LOG(debug) << "NState::load " << path ;  

    int rc = BMap<std::string, std::string>::load(&m_kv, m_dir, filename.c_str(), depth ) ;
    return rc  ; 
}


void NState::roundtrip()
{
    save();
    load();
}




