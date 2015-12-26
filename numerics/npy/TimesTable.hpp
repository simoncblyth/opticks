#pragma once

#include <vector>
#include <string>

class Times ; 

class TimesTable {
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


inline TimesTable::TimesTable(const std::vector<std::string>& columns)
{
    init(columns);
}

inline Times* TimesTable::getColumn(unsigned int j)
{
    return m_table[j] ; 
}
inline std::vector<std::string>& TimesTable::getLines()
{
    makeLines(); 
    return m_lines ;  
}


 
