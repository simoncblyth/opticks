#pragma once

#include <vector>
#include <string>
#include "NGLMExt.hpp"

/*
In general the intersection between sub-objects frontier will not be a plane, 
nevertherless finding the best fit plane provides a way to simplify 
handling : can order loop edges and frontier points 
according to an angle after picking some basis vector that lies in the plane.

Hmm project spoke vectors from the cog onto the plane...
 


* http://www.ilikebigbits.com/blog/2015/3/2/plane-from-points

    Edit: as the commenter Paul pointed out, this method will minimize the squares
    of the residuals as perpendicular to the main axis, not the residuals
    perpendicular to the plane. If the residuals are small (i.e. your points all
    lie close to the resulting plane), then this method will probably suffice.
    However, if your points are more spread then this method may not be the best
    fit.

*/

struct NPY_API NPlaneFromPoints
{
    NPlaneFromPoints(unsigned reference=0) 
       :
        reference(reference),
        cog(0,0,0),
        det(0,0,0),
        nrm(0,0,0),
        xx(0),
        xy(0),
        xz(0),
        yy(0),
        yz(0),
        zz(0)
    {
    }

    void add(float x, float y, float z);
    void add(const glm::vec3& p);

    void update();
    void update_cog();
    void update_nrm();
    void update_projection();

    void project_into_plane( glm::vec3& proj, const glm::vec3& p) const ;
    float azimuthal_diff(unsigned j) const ;
    float azimuthal_diff(const glm::vec3& p) const ;

    std::string desc() const ;
    void dump(const char* msg="NPlaneFromPoints::dump") const ;


    unsigned reference ; 
    std::vector<glm::vec3> points ; 
    std::vector<glm::vec3> projection ; 

    glm::vec3 cog ; 
    glm::vec3 det ; 
    glm::vec3 nrm ; 

    float xx ;
    float xy ;
    float xz ;

    float yy ;
    float yz ;
    float zz ;
};


