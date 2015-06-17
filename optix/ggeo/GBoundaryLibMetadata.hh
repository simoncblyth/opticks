#pragma once

#include <string>
#include <boost/property_tree/ptree.hpp>

#include "GPropertyMap.hh"

// the canonical instance of GBoundaryLibMetadata 
// is created by GBoundaryLib::createWavelengthBuffer
// and is saved into the same cache directory as the wavelength buffer


class GBoundaryLibMetadata {
  public:
      static const char* filename ; 
      static const char* mapname ; 

      GBoundaryLibMetadata();

  public:
      static GBoundaryLibMetadata* load(const char* dir);
      void save(const char* dir);
      void Summary(const char* msg);
      void createMaterialMap();

   public:
      // 1-based substance code to allow cos_theta sign flipped boundary code, zero corresponds to "nohit"
      unsigned int getBoundaryCode(unsigned int isub);
      std::string getBoundaryName(unsigned int isub);
      unsigned int getNumBoundary();
      void dumpNames();
      std::map<int, std::string> getBoundaryNames();

   public:
      std::string getBoundaryQty(unsigned int isub, const char* cat, const char* key);
      std::string getBoundaryQtyByIndex(unsigned int isub, unsigned int icat, const char* tag);

   public:
      std::string get(const char* kfmt, const char* idx);
      std::string get(const char* kfmt, unsigned int idx);
      void add(const char* kfmt, unsigned int isub, const char* cat, GPropertyMap<float>* pmap );
      void addDigest(const char* kfmt, unsigned int isub, const char* cat, const char* dig );
      void addMaterial(unsigned int isub, const char* cat, const char* shortname, const char* digest );

  private:
      void read(const char* path);
      void add(const char* kfmt, unsigned int isub, const char* cat, const char* tag, const char* val);

  private:
      boost::property_tree::ptree   m_tree;
      boost::property_tree::ptree   m_material_map ;

};

