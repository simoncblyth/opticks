#pragma once

#include <vector>
#include <string>
#include <map>
#include "string.h"

class Times {
  public:
     typedef std::pair<std::string, double>  SD ; 
     typedef std::vector<SD>                VSD ; 
  public:
     static void compare(const std::vector<Times*>&, unsigned int nwid=25, unsigned int twid=10, unsigned int tprec=3);
     static std::string name(const char* type, const char* tag);
  public:
     Times();
     Times* clone();
     void add(const char* name, double t );
  public:
     void setScale(double s);
     double getScale(); 
     void setLabel(const char* label);
     const char* getLabel();
  public:
     void dump(const char* msg="Times::dump");
     unsigned int getSize();
     std::vector<std::pair<std::string, double> >& getTimes();
 
  public:
     static Times* load(const char* dir, const char* name);
     void save(const char* dir, const char* name);
  private:
     void load_(const char* dir, const char* name);
  private:
     VSD         m_times ;  
     double      m_scale ; 
     const char* m_label ; 

};

inline Times::Times()  : m_scale(1.0) , m_label(NULL) 
{
}

inline Times* Times::clone()
{
    Times* ts = new Times ; 
    for(VSD::const_iterator it=m_times.begin() ; it != m_times.end() ; it++) ts->add(it->first.c_str(), it->second) ;
    return ts ; 
}


inline void Times::add(const char* name, double t )
{
    m_times.push_back(SD(name, t));
}
inline unsigned int Times::getSize()
{
    return m_times.size();
}
inline std::vector<std::pair<std::string, double> >& Times::getTimes()
{
    return m_times ;
}
inline double Times::getScale()
{
    return m_scale ; 
}
inline void Times::setScale(double scale)
{
    m_scale = scale  ; 
}


inline void Times::setLabel(const char* label)
{
    m_label = strdup(label);
}
inline const char* Times::getLabel()
{
    return m_label ; 
}

