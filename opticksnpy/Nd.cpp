#include <iomanip>
#include <sstream>
#include <cstring>
#include <algorithm>

#include "SDigest.hh"

#include "BFile.hh"
#include "BStr.hh"

#include "Nd.hpp"
#include "GLMFormat.hpp"
#include "NGLMExt.hpp"



std::string nd::pvtag(unsigned nch) const 
{
    std::string pvn = BFile::Name(pvname.c_str()) ;
    std::string pvns = BStr::firstChars(pvn.c_str(), nch);  
    return pvns ; 
}


std::string nd::desc()
{
    std::stringstream ss ; 

    ss << "nd"
       << " idx/repeatIdx/mesh/nch/depth/nprog "
       << " [" 
       << std::setw(3) << idx 
       << ":" 
       << std::setw(3) << repeatIdx
       << ":" 
       << std::setw(3) << mesh 
       << ":" 
       << std::setw(3) << children.size() 
       << ":" 
       << std::setw(2) << depth
       << ":" 
       << std::setw(4) << _progeny.size()
       << "]"
       << " bnd:" << boundary 
       ;

    return ss.str();
}

std::string nd::detail()
{
    std::stringstream ss ; 
    ss << desc() << std::endl ;

    if(transform)  ss << gpresent("nd.tr.t",  transform->t ) << std::endl ;  
    if(gtransform) ss << gpresent("nd.gtr.t", gtransform->t ) << std::endl ;  
 
    return ss.str();
}


nmat4triple* nd::make_global_transform(nd* n)
{
    std::vector<nmat4triple*> tvq ; 
    while(n)
    {
        if(n->transform) tvq.push_back(n->transform);
        n = n->parent ; 
    }
    bool reverse = true ; // as tvq in leaf-to-root order
    return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ; 
}



void nd::_collect_ancestors(nd* n, std::vector<nd*>& ancestors)
{
    while(n)
    {
        ancestors.push_back(n);
        n = n->parent ; 
    }
    std::reverse( ancestors.begin(), ancestors.end() );
}
const std::vector<nd*>& nd::get_ancestors()
{
    if(_ancestors.size() == 0) _collect_ancestors(parent, _ancestors ); 
    return _ancestors  ; 
}


const std::vector<nd*>& nd::get_progeny()
{
    if(_progeny.size() == 0)
    {
        _collect_progeny_r(this, _progeny, 0); 
    }
    return _progeny ; 
}


unsigned nd::get_progeny_count()
{
    const std::vector<nd*>& progs = get_progeny();
    return progs.size();
}



void nd::_collect_progeny_r(nd* n, std::vector<nd*>& progeny, int depth)
{
    if(depth > 0) progeny.push_back(n);  // exclude depth=0 avoids collecting self 
    for(unsigned i = 0; i < n->children.size(); i++) _collect_progeny_r(n->children[i], progeny, depth+1);
}

std::string nd::_mesh_id() const  
{
    std::stringstream ss ; 
    ss << mesh ; 
    return ss.str();
}

std::string nd::_make_mesh_digest() const 
{
    SDigest dig ;

    std::string mid = _mesh_id() ;
    dig.update(mid);

    return dig.finalize();
}

std::string nd::_make_local_digest() const 
{
    SDigest dig ;

    std::string tdig = transform ? transform->digest() : "" ;
    dig.update(tdig);

    std::string mid = _mesh_id() ;
    dig.update(mid);

    return dig.finalize();
}

std::string nd::_make_digest(const std::vector<nd*>& nds, nd* extra)
{
    SDigest dig ;

    for(unsigned i=0 ; i < nds.size() ; i++)
    {
        nd* n = nds[i];
        const std::string& ndig = n->get_local_digest();
        dig.update(ndig);
    }

    if(extra)
    {
        // following GNode::localDigest pattern of not including the transform 
        // with the extra self digest
        // ... is that appropriate ?

        const std::string& xdig = extra->get_mesh_digest();
        dig.update( xdig );
    }
    return dig.finalize();
}



const std::string& nd::get_local_digest()
{
    if(_local_digest.empty()) _local_digest = _make_local_digest();
    return _local_digest ;
}

const std::string& nd::get_mesh_digest()
{
    if(_mesh_digest.empty()) _mesh_digest = _make_mesh_digest();
    return _mesh_digest ;
}


const std::string& nd::get_progeny_digest()
{
    if(_progeny_digest.empty())
    {
        const std::vector<nd*>& progeny = get_progeny();
        nd* extra = this ;
        _progeny_digest = _make_digest(progeny, extra) ;
    }
    return _progeny_digest ;
}

bool nd::has_progeny_digest(const std::string& dig)
{
    const std::string& pdig = get_progeny_digest();
    return strcmp(pdig.c_str(), dig.c_str())==0 ;
}


void nd::_collect_nodes_r(std::vector<nd*>& selection, const std::string& pdig)
{
    if(has_progeny_digest(pdig))
    {
        selection.push_back(this);
    }
    else   // NB: does not traverse into matching nodes 
    {
        for(unsigned i = 0; i < children.size(); i++) children[i]->_collect_nodes_r(selection, pdig );
    }
}

std::vector<nd*> nd::find_nodes(const std::string& pdig)
{
    std::vector<nd*> selection ;
    _collect_nodes_r(selection, pdig );
    return selection ;
}



nd* nd::_find_node_r(const std::string& pdig)
{
    nd* n = NULL ; 
    if(has_progeny_digest(pdig))
    {
        n = this ; 
    }
    else
    {
        for(unsigned i = 0; i < children.size(); i++) 
        { 
             n = children[i]->_find_node_r(pdig);
             if(n) break ; 
        }
    }
    return n ; 
}

nd* nd::find_node(const std::string& pdig)
{
    return _find_node_r(pdig);
}





