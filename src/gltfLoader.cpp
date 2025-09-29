// gltfLoader.cpp — the ONLY TU that builds tinygltf
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"           // include once, here
#include "gltfLoader.h"          // your header

#include <iostream>
#include <cstring>

// ---------------- Implementation ----------------

bool glTFLoader::loadModel(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;





    bool ok = false;
    // Pick loader by file extension; also tolerates .gltf ASCII
    if (filename.size() >= 4 &&
        (filename.size() > 4 && (filename.substr(filename.size() - 4) == ".glb" ||
            filename.substr(filename.size() - 4) == ".GLB"))) {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    }
    else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    }

    if (!warn.empty()) std::cout << "[tinygltf warn] " << warn << "\n";
    if (!err.empty())  std::cout << "[tinygltf err ] " << err << "\n";

    if (!ok) {
        std::cout << "Failed to load glTF file: " << filename << "\n";
        return false;
    }

    meshes_.clear();
    processModel(model);
    return true;
}

void glTFLoader::processModel(const tinygltf::Model& model) {
    // Iterate meshes / primitives and pull POSITION + indices
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            Mesh out;

            // POSITION (required)
            auto it = prim.attributes.find("POSITION");
            if (it != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[it->second];
                const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
                const tinygltf::Buffer& buf = model.buffers[bv.buffer];

                // expect VEC3 float
                if (acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || acc.type != TINYGLTF_TYPE_VEC3) {
                    std::cout << "Unsupported POSITION accessor format\n";
                }
                else {
                    const size_t offs = bv.byteOffset + acc.byteOffset;
                    const float* ptr = reinterpret_cast<const float*>(&buf.data[offs]);
                    out.positions.assign(ptr, ptr + acc.count * 3);
                }
            }

            // Indices (optional)
            if (prim.indices >= 0) {
                const tinygltf::Accessor& acc = model.accessors[prim.indices];
                const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
                const tinygltf::Buffer& buf = model.buffers[bv.buffer];
                const size_t offs = bv.byteOffset + acc.byteOffset;

                out.indices.reserve(acc.count);
                switch (acc.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const uint16_t* idx = reinterpret_cast<const uint16_t*>(&buf.data[offs]);
                    for (size_t i = 0; i < acc.count; ++i) out.indices.push_back(static_cast<uint32_t>(idx[i]));
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const uint32_t* idx = reinterpret_cast<const uint32_t*>(&buf.data[offs]);
                    out.indices.assign(idx, idx + acc.count);
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    const uint8_t* idx = reinterpret_cast<const uint8_t*>(&buf.data[offs]);
                    for (size_t i = 0; i < acc.count; ++i) out.indices.push_back(static_cast<uint32_t>(idx[i]));
                    break;
                }
                default:
                    std::cout << "Unsupported index component type\n";
                    break;
                }
            }
            else {
                // no index accessor => generate 0..N-1 (triangulated required by spec in practice)
                const size_t vtxCount = out.positions.size() / 3;
                out.indices.resize(vtxCount);
                for (size_t i = 0; i < vtxCount; ++i) out.indices[i] = static_cast<uint32_t>(i);
            }

            meshes_.push_back(std::move(out));
        }
    }
}

std::vector<MeshTriangle> glTFLoader::getTriangles() const {
    std::vector<MeshTriangle> tris;
    for (const auto& m : meshes_) {
        // assume triangles
        for (size_t i = 0; i + 2 < m.indices.size(); i += 3) {
            MeshTriangle t;
            t.v0 = getVertex(m, m.indices[i + 0]);
            t.v1 = getVertex(m, m.indices[i + 1]);
            t.v2 = getVertex(m, m.indices[i + 2]);
            tris.push_back(t);
        }
    }
    return tris;
}
