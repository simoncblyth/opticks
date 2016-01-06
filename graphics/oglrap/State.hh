#pragma once

#include <map>
#include <cstring>
#include <string>
#include <vector>

// think first about saving and restoring state of a collection of objects
// only after that works worry about bookmark-like jumping between states

class Configurable ; 

class State {
   public:
       State(const char* dir="/tmp", const char* name="state");
       void addConfigurable(Configurable* configurable);
   public:
       void gui();
       void roundtrip();
       void save();
       void load();
       void collect();
       void apply();
   private:
       Configurable* getConfigurable(const char* prefix); 
       std::string get(const char* key);
       void set(const char* key, const char* val);
       std::string getFileName();
       std::string formKey(const char* prefix, const char* tag);
       void splitKey(std::vector<std::string>& prefix_tag, const char* key);

       void apply(const char* k, const char* v);
       unsigned int collect(Configurable* configurable);
       void setNumChanges(unsigned int num_changes);
       unsigned int getNumChanges();
   private:
       const char*                           m_dir ; 
       const char*                           m_name ; 
       unsigned int                          m_num_changes ; 
       std::map<std::string, std::string>    m_kv ; 
       std::map<std::string, Configurable*>  m_configurables ; 

};

inline State::State(const char* dir, const char* name) 
    :
    m_dir(strdup(dir)),
    m_name(strdup(name)),
    m_num_changes(0)
{
}

inline void State::setNumChanges(unsigned int num_changes)
{
    m_num_changes = num_changes ; 
}
inline unsigned int State::getNumChanges()
{
    return m_num_changes ; 
}
