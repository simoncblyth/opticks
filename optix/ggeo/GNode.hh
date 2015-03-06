#ifndef GNODE_H
#define GNODE_H

#include <vector>

class GNode {
  public:
      GNode(unsigned int index);
      virtual ~GNode();

  public: 
      void Summary(const char* msg="GNode::Summary");

  public:
      void setParent(GNode* parent);
      void addChild(GNode* child);
      void setDescription(char* desc);

  public:
      unsigned int getIndex();
      GNode* getParent(); 
      GNode* getChild(unsigned int index);
      unsigned int getNumChildren();
      char* getDescription();

  private:
      unsigned int m_index ; 
      GNode* m_parent ; 
      std::vector<GNode*> m_children ;
      char* m_description ;

};


#endif
