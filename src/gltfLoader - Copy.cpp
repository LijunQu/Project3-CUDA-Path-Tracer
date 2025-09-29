#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
//#include "tiny_gltf.h"


#include "glTFLoader.h"

//#ifndef TINYGLTF_IMPLEMENTATION
//#define TINYGLTF_IMPLEMENTATION
//#endif
#include <tiny_gltf.h>
#include <iostream>

bool glTFLoader::loadModel(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        printf("Warning: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Error: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to load glTF file: %s\n", filename.c_str());
        return false;
    }

    processModel(model);
    return true;
}

void glTFLoader::processModel(const tinygltf::Model& model) {
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            Mesh newMesh;

            // Extract position data
            if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
                const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("POSITION")];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                newMesh.positions.assign(positions, positions + accessor.count * 3);
            }

            // Extract index data
            if (primitive.indices >= 0) {
                const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                newMesh.indices.reserve(accessor.count);
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_gltf.h>
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
                switch (accessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const uint16_t* indices = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                    for (size_t i = 0; i < accessor.count; ++i) {
                        newMesh.indices.push_back(static_cast<uint32_t>(indices[i]));
                    }
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const uint32_t* indices = reinterpret_cast<const uint32_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                    newMesh.indices.assign(indices, indices + accessor.count);
                    break;
                }
                default:
                    printf("Unsupported index component type\n");
                    break;
                }
            }

            meshes.push_back(newMesh);
        }
    }
}