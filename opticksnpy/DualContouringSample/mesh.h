#pragma once
#include <vector>
#include <glm/glm.hpp>

// ----------------------------------------------------------------------------

struct MeshVertex
{
	MeshVertex(const glm::vec3& _xyz, const glm::vec3& _normal)
		: xyz(_xyz)
		, normal(_normal)
	{
	}

	glm::vec3		xyz, normal;
};

typedef std::vector<MeshVertex> VertexBuffer;
typedef std::vector<int> IndexBuffer;


