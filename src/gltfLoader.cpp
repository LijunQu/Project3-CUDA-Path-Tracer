// gltfLoader.cpp the ONLY TU that builds tinygltf
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "gltfLoader.h" 

#include <iostream>
#include <cstring>

// ---------------- Implementation ----------------



bool glTFLoader::loadModel(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;


    // ---- Ignore image loading (we only need geometry) ----
    auto IgnoreImageLoader = +[](tinygltf::Image*,
        const int,
        std::string*,
        std::string*,
        int, int,
        const unsigned char*,
        int,
        void*) -> bool {
            return true;
        };
    loader.SetImageLoader(IgnoreImageLoader, nullptr);



    bool ok = false;
    // Pick loader by file extension; also tolerates .gltf ASCII
    if (filename.size() >= 4 &&
        (filename.size() > 4 && (filename.substr(filename.size() - 4) == ".glb" ||
            filename.substr(filename.size() - 4) == ".GLB"))) {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, filename);

        // ... choose ASCII vs Binary, then:
        if (!warn.empty()) std::cout << "[tinygltf warn] " << warn << "\n";
        if (!err.empty())  std::cout << "[tinygltf err ] " << err << "\n";

        std::cout << "[GLTF-LOAD] ok=" << ok << " path=" << filename << "\n";

        if (!ok) {
            std::cout << "Failed to load glTF file: " << filename << "\n";
            return false;
        }

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
    model_ = model;
    processModel(model_);
    return true;
}

void glTFLoader::processModel(const tinygltf::Model& model) {
    // Iterate meshes / primitives and pull POSITION + indices
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            Mesh out;


            out.gltfMaterialIndex = prim.material;

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

            // Load TEXCOORD_0 (UV coordinates)
            auto uvIt = prim.attributes.find("TEXCOORD_0");
            if (uvIt != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[uvIt->second];
                const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
                const tinygltf::Buffer& buf = model.buffers[bv.buffer];

                if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
                    acc.type == TINYGLTF_TYPE_VEC2) {
                    const size_t offs = bv.byteOffset + acc.byteOffset;
                    const float* ptr = reinterpret_cast<const float*>(&buf.data[offs]);
                    out.uvs.assign(ptr, ptr + acc.count * 2);
                    //std::cout << "Loaded " << acc.count << " UV coordinates\n";
                }
                else {
                    std::cout << "Unsupported TEXCOORD format\n";
                }
            }
            else {
                std::cout << "No UV coordinates found in mesh\n";
            }

            // Indices
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
        bool hasUVs = !m.uvs.empty();
        // assume triangles
        for (size_t i = 0; i + 2 < m.indices.size(); i += 3) {
            MeshTriangle t;
            t.v0 = getVertex(m, m.indices[i + 0]);
            t.v1 = getVertex(m, m.indices[i + 1]);
            t.v2 = getVertex(m, m.indices[i + 2]);


            if (hasUVs) {
                t.uv0 = getUV(m, m.indices[i + 0]);
                t.uv1 = getUV(m, m.indices[i + 1]);
                t.uv2 = getUV(m, m.indices[i + 2]);
            }
            else {
                t.uv0 = glm::vec2(0.0f, 0.0f);
                t.uv1 = glm::vec2(1.0f, 0.0f);
                t.uv2 = glm::vec2(0.0f, 1.0f);
            }

            t.gltfMaterialIndex = m.gltfMaterialIndex;
            tris.push_back(t);
        }
    }
    return tris;
}


int glTFLoader::getMaterialCount() const {
    return model_.materials.size();
}


bool glTFLoader::getMaterialTextures(int matIdx, std::string& baseColorUri, std::string& normalUri) const {
    if (matIdx < 0 || matIdx >= model_.materials.size()) return false;

    const auto& mat = model_.materials[matIdx];

    baseColorUri.clear();
    normalUri.clear();

    // Get base color texture
    int baseTexIdx = mat.pbrMetallicRoughness.baseColorTexture.index;
    if (baseTexIdx >= 0 && baseTexIdx < model_.textures.size()) {
        const auto& tex = model_.textures[baseTexIdx];
        if (tex.source >= 0 && tex.source < model_.images.size()) {
            baseColorUri = model_.images[tex.source].uri;
        }
    }

    // Get normal texture
    int normTexIdx = mat.normalTexture.index;
    if (normTexIdx >= 0 && normTexIdx < model_.textures.size()) {
        const auto& tex = model_.textures[normTexIdx];
        if (tex.source >= 0 && tex.source < model_.images.size()) {
            normalUri = model_.images[tex.source].uri;
        }
    }

    return true;
}