#pragma once
#include <cstring>


class NGunConfig {
   public:
       NGunConfig(const char* config);
   private:
       void init();
   private:
       const char* m_config ;

};

inline NGunConfig::NGunConfig(const char* config)
    :
     m_config( config ? strdup(config) : NULL )
{
    init();
}

