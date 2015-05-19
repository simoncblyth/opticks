#pragma once

class Configurable {
    public: 
       virtual ~Configurable(){};

       static bool accepts(const char* name);
       void configure(const char* name, const char* value);
       virtual std::vector<std::string> getTags() = 0 ;
       virtual std::string get(const char* name) = 0 ;
       virtual void set(const char* name, std::string& val) = 0 ;

};
