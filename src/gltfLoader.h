#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <cstdint>

// Forward-declare, so this header does NOT pull tiny_gltf.h into other files.
namespace tinygltf { class Model; }

struct MeshTriangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

class glTFLoader {
public:
    struct Mesh {
        std::vector<float>   positions;  // x y z packed
        std::vector<uint32_t> indices;   // triangle indices (0-based)
    };

    glTFLoader() = default;

    // Load a GLTF/GLB file. On success, fills `meshes_`.
    bool loadModel(const std::string& filename);

    // Build triangles from all meshes (positions + indices).
    std::vector<MeshTriangle> getTriangles() const;

private:
    std::vector<Mesh> meshes_;

    glm::vec3 getVertex(const Mesh& mesh, uint32_t index) const {
        const size_t i = static_cast<size_t>(index) * 3;
        return { mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2] };
    }

    // Defined in the .cpp (needs tiny_gltf.h there, not here)
    void processModel(const tinygltf::Model& model);
};
