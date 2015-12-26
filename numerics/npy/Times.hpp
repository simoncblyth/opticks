#pragma once

#include <vector>
#include <string>
#include <map>
#include <cstring>

class Times {
  public:
     typedef std::pair<std::string, double>  SD ; 
     typedef std::vector<SD>                VSD ; 
  public:
     static void compare(const std::vector<Times*>&, unsigned int nwid=25, unsigned int twid=10, unsigned int tprec=3);
     static std::string name(const char* type, const char* tag);
     std::string name();
  public:
     Times(const char* label="nolabel");
     void setLabel(const char* label);
     Times* clone(const char* label);
     void add(const char* name, double t );
     unsigned int getNumEntries();
  public:
     void setScale(double s);
     double getScale(); 
     const char* getLabel();
  public:
     void dump(const char* msg="Times::dump");
     unsigned int getSize();
     std::vector<std::pair<std::string, double> >& getTimes();
     std::pair<std::string, double>&  getEntry(unsigned int i);
  public:
     void save(const char* dir);
     void load(const char* dir);
     void load(const char* dir, const char* name);
  public:
     static Times* load(const char* label, const char* dir, const char* name);
  private:
     VSD         m_times ;  
     double      m_scale ; 
     const char* m_label ; 

};

inline Times::Times(const char* label)  
   : 
     m_scale(1.0) , 
     m_label(strdup(label)) 
{
}

inline void Times::setLabel(const char* label)
{
    m_label = strdup(label);
}

inline Times* Times::clone(const char* label)
{
    Times* ts = new Times(label) ; 
    for(VSD::const_iterator it=m_times.begin() ; it != m_times.end() ; it++) ts->add(it->first.c_str(), it->second) ;
    return ts ; 
}

inline unsigned int Times::getNumEntries()
{
    return m_times.size();
}
inline std::pair<std::string, double>&  Times::getEntry(unsigned int i)
{
    return m_times[i] ;
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


inline const char* Times::getLabel()
{
    return m_label ; 
}

