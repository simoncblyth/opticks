#pragma once

#include <vector>
#include <string>
#include <map>
#include <cstring>

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API Times {
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

#include "NPY_TAIL.hh"


