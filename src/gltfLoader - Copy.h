#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "tiny_gltf.h"

struct MeshTriangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

class glTFLoader {
public:

    struct Mesh {
        std::vector<float> positions; // Stores x, y, z coordinates consecutively
        std::vector<uint32_t> indices;
    };

    glTFLoader() {}

    bool loadModel(const std::string& filename);

    std::vector<MeshTriangle> getTriangles() const {
        std::vector<MeshTriangle> triangles;
        for (const auto& mesh : meshes) {
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                MeshTriangle tri;
                tri.v0 = getVertex(mesh, mesh.indices[i]);
                tri.v1 = getVertex(mesh, mesh.indices[i + 1]);
                tri.v2 = getVertex(mesh, mesh.indices[i + 2]);
                triangles.push_back(tri);
            }
        }
        return triangles;
    }

private:
    std::vector<Mesh> meshes;

    glm::vec3 getVertex(const Mesh& mesh, uint32_t index) const {
        size_t i = index * 3;
        return { mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2] };
    }

    void processModel(const tinygltf::Model& model);
};