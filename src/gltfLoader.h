#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include "sceneStructs.h"
#include "tiny_gltf.h" 

// Forward-declare, so this header does NOT pull tiny_gltf.h into other files.
namespace tinygltf { class Model; }

struct MeshTriangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 centroid;
    glm::vec2 uv0, uv1, uv2;
    int materialId;

    glm::vec3 edge1, edge2;
    glm::vec2 deltaUV1, deltaUV2;
    int gltfMaterialIndex = -1;
};

class glTFLoader {
public:
    struct Mesh {
        std::vector<float>   positions;  // x y z packed
        std::vector<uint32_t> indices;
        std::vector<float> uvs;
        int gltfMaterialIndex = -1;

        glm::mat4 nodeTransform = glm::mat4(1.0f);
    };

    glTFLoader() = default;

    // Load a GLTF/GLB file. On success, fills `meshes_`.
    bool loadModel(const std::string& filename);

    // Build triangles from all meshes (positions + indices).
    std::vector<MeshTriangle> getTriangles() const;

    int glTFLoader::getMaterialCount() const;

    bool glTFLoader::getMaterialTextures(int matIdx, std::string& baseColorUri, std::string& normalUri,
        std::string& metallicRoughnessUri, float& metallic, float& roughness) const;

private:
    std::vector<Mesh> meshes_;
    tinygltf::Model model_;

    glm::vec3 getVertex(const Mesh& mesh, uint32_t index) const {
        const size_t i = static_cast<size_t>(index) * 3;
        return { mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2] };
    }

    void processModel(const tinygltf::Model& model);
    void glTFLoader::processNode(const tinygltf::Model& model, int nodeIdx, const glm::mat4& parentTransform);

    glm::vec2 getUV(const Mesh& mesh, uint32_t index) const {
        if (mesh.uvs.empty()) return glm::vec2(0.0f);
        const size_t i = static_cast<size_t>(index) * 2;
        return { mesh.uvs[i], mesh.uvs[i + 1] };
    }
};