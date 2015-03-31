#ifndef CONFIG_H
#define CONFIG_H

class Config {
   public:
       Config();
       virtual ~Config();
       unsigned int getUdpPort();
       void setUdpPort(unsigned int udpPort);
   private:
       unsigned int m_udpPort ;

};

#endif
