#include "State.hh"
#include "Configurable.hh"

#include <sstream>
#include <boost/algorithm/string.hpp>

// npy-
#include "jsonutil.hpp"
#include "NLog.hpp"

#ifdef GUI_
#include <imgui.h>
#endif




void State::addConfigurable(Configurable* configurable)
{
    const char* prefix = configurable->getPrefix();
    m_configurables[prefix] = configurable ; 
}

Configurable* State::getConfigurable(const char* prefix)
{
    return m_configurables.count(prefix) == 1 ? m_configurables[prefix] : NULL ;
}

std::string State::getFileName()
{
   std::stringstream ss ; 
   ss << m_name << ".ini" ; 
   return ss.str(); 
}
void State::save()
{
    std::string filename = getFileName();
    saveMap<std::string, std::string>(m_kv, m_dir, filename.c_str() ) ;
}
void State::load()
{
    std::string filename = getFileName();
    loadMap<std::string, std::string>(m_kv, m_dir, filename.c_str() ) ;
}


std::string State::get(const char* key)
{
    std::string empty ;
    return m_kv.count(key) > 0 ? m_kv[key] : empty ; 
}

void State::set(const char* key, const char* val)
{
    m_kv[key] = val ; 
}

void State::collect()
{
    unsigned int changes(0);
    typedef std::map<std::string, Configurable*>::const_iterator SCI ; 
    for(SCI it=m_configurables.begin() ; it != m_configurables.end() ; it++)
    {
       changes += collect(it->second) ;
    }
    setNumChanges(changes) ; 
}


std::string State::formKey(const char* prefix, const char* tag)
{
    char key[64];
    snprintf(key, 64, "%s.%s", prefix, tag );
    return key ;  
}

void State::splitKey(std::vector<std::string>& prefix_tag, const char* key)
{
    boost::split(prefix_tag, key, boost::is_any_of(".")); 
    assert(prefix_tag.size() == 2 && "State::splitKey malformed key, expecting form \"prefix.tag\"" ); 
}


unsigned int State::collect(Configurable* configurable)
{
    // Configurable is an abstract get/set/getTags/accepts/configure protocol 
    if(!configurable) return 0 ; 

    std::string empty ;
    std::vector<std::string> tags = configurable->getTags();

    const char* prefix = configurable->getPrefix();
 
    LOG(info) << "State::collect" 
              << " prefix " << prefix 
              ; 

    unsigned int changes(0);

    for(unsigned int i=0 ; i < tags.size(); i++)
    {
        const char* tag = tags[i].c_str();

        std::string key = formKey(prefix, tag);
        std::string val = configurable->get(tag);
        std::string prior = get(key.c_str());
        
        LOG(info) << "State::collect"
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
            LOG(info) << "State::collect"
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


void State::apply()
{
    typedef std::map<std::string, std::string>::const_iterator SSI ; 
    for(SSI it=m_kv.begin() ; it != m_kv.end() ; it++)
    {
         std::string key = it->first ;           
         std::string val = it->second ;

         std::vector<std::string> prefix_tag ; 
         splitKey(prefix_tag, key.c_str());

         std::string prefix = prefix_tag[0] ; 
         std::string tag = prefix_tag[1] ; 

         Configurable* configurable = getConfigurable(prefix.c_str());
         if(configurable)
            configurable->configure(tag.c_str(), val.c_str());
         else
            LOG(warning) << "State::apply no configurable for prefix " << prefix ;  
    }
}






void State::gui()
{
#ifdef GUI_

    if(ImGui::Button("roundtrip")) roundtrip();
    ImGui::SameLine();
    if(ImGui::Button("save")) save();
    ImGui::SameLine();
    if(ImGui::Button("load")) load();
    ImGui::SameLine();
    if(ImGui::Button("collect")) collect();
    ImGui::SameLine();
    if(ImGui::Button("apply")) apply();

#endif
}



void State::roundtrip()
{
    LOG(info) << "State::roundtrip " ; 
    save();
    load();
}






