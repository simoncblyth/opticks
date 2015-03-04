#ifndef GNODE_H
#define GNODE_H

#include <vector>

class GSolid ;

class GNode {
  public:
      GNode();
      virtual ~GNode();

  public: 
      void Summary(const char* msg="GNode::Summary");

  public: 
      unsigned int getNumChildren();
      GNode* getChild(unsigned int n);

  private:
      GSolid* m_solid ; 

      std::vector<GNode*> m_children ;

};


#endif
