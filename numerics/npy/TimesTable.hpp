#pragma once

#include <vector>
#include <string>

class Times ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API TimesTable {
    public:
        TimesTable(const char* columns, const char* delim=","); 
        TimesTable(const std::vector<std::string>& columns);
        void dump(const char* msg="TimesTable::dump");
        Times* getColumn(unsigned int j);
        void makeLines();
        std::vector<std::string>& getLines(); 
    public:
        void save(const char* dir);
        void load(const char* dir);
    private:
        void init(const std::vector<std::string>& columns);
    private:
        std::vector<Times*>      m_table ; 
        std::vector<std::string> m_lines ; 
};

#include "NPY_TAIL.hh"




 
