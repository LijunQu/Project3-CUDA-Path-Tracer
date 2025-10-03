#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#define STB_IMAGE_IMPLEMENTATION


using namespace std;
using json = nlohmann::json;

Texture Scene::loadTexture(const std::string& filepath) {
    Texture tex;
    int channels;

    std::cout << "Loading texture: " << filepath << "\n";
    unsigned char* data = stbi_load(filepath.c_str(), &tex.width, &tex.height, &channels, 3);

    if (!data) {
        std::cout << "Failed to load texture: " << filepath << "\n";
        tex.width = tex.height = 0;
        tex.data = nullptr;
        return tex;
    }

    std::cout << "Loaded texture: " << tex.width << "x" << tex.height << " (" << channels << " channels)\n";

    // Allocate host memory and convert to float RGB
    size_t pixelCount = tex.width * tex.height;
    glm::vec3* hostData = new glm::vec3[pixelCount];

    for (size_t i = 0; i < pixelCount; i++) {
        hostData[i] = glm::vec3(
            data[i * 3 + 0] / 255.0f,
            data[i * 3 + 1] / 255.0f,
            data[i * 3 + 2] / 255.0f
        );
    }

    tex.data = hostData;

    stbi_image_free(data);
    return tex;
}

std::vector<MeshTriangle> Scene::getTriangleBuffer()
{
    if (!jsonLoadedNonCuda)
    {
        std::cout << "loadJSON not called before CUDA load mesh!\n";
        //return;
        exit(EXIT_FAILURE);
    }
    std::ifstream f(jsonName_str);
    json data = json::parse(f);

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "mesh")
        {
            return triangles;
        }
    }
    return std::vector<MeshTriangle>();
}



using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    jsonLoadedNonCuda = true;
    jsonName_str = jsonName;
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            if (p.contains("TEXTURE")) {
                std::string texPath = p["TEXTURE"];
                newMaterial.textureID = textures.size();
                newMaterial.hasTexture = true;
                textures.push_back(loadTexture(texPath));
            }
            else {
                newMaterial.textureID = -1;
                newMaterial.hasTexture = false;
            }

            // ADD NORMAL MAP LOADING
            if (p.contains("NORMALMAP")) {
                std::string normalPath = p["NORMALMAP"];
                newMaterial.normalMapID = textures.size();
                newMaterial.hasNormalMap = true;
                textures.push_back(loadTexture(normalPath));
                std::cout << "Loaded normal map for material: " << name << "\n";
            }
            else {
                newMaterial.normalMapID = -1;
                newMaterial.hasNormalMap = false;
            }

        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.hasReflective = 1.0f;
            newMaterial.indexOfRefraction = 1.0f / 1.55f;
        }
        else if (p["TYPE"] == "Transparent") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = 1.0f / 1.55f;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;


        if (type == "mesh")
        {

            std::cout << "Loading mesh from: " << p["FILEPATH"] << "\n";

            glTFLoader loader;
            const auto& filePath = p["FILEPATH"];

            if (!loader.loadModel(filePath)) {
                std::cout << "Error loading gltf model!\n";
                exit(EXIT_FAILURE);
            }

            std::vector<MeshTriangle> newTriangles = loader.getTriangles();
            std::cout << "Loaded " << newTriangles.size() << " triangles\n";

            int startIdx = triangles.size();
            //std::cout << "Current triangle count: " << startIdx << "\n";







            for (int i = 0; i < newTriangles.size(); i++) {
                Geom newGeom;
                newGeom.type = TRI;

                newGeom.triangle_index = i;

                newGeom.materialid = MatNameToID[p["MATERIAL"]];

                const auto& trans = p["TRANS"];
                const auto& rotat = p["ROTAT"];
                const auto& scale = p["SCALE"];
                newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
                newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);

                newGeom.transform = utilityCore::buildTransformationMatrix(
                    newGeom.translation, newGeom.rotation, newGeom.scale);

                MeshTriangle transformedTri;
                transformedTri.v0 = glm::vec3(newGeom.transform * glm::vec4(newTriangles[i].v0, 1.0f));
                transformedTri.v1 = glm::vec3(newGeom.transform * glm::vec4(newTriangles[i].v1, 1.0f));
                transformedTri.v2 = glm::vec3(newGeom.transform * glm::vec4(newTriangles[i].v2, 1.0f));
                
                transformedTri.uv0 = newTriangles[i].uv0;
                transformedTri.uv1 = newTriangles[i].uv1;
                transformedTri.uv2 = newTriangles[i].uv2;

                transformedTri.edge1 = transformedTri.v1 - transformedTri.v0;
                transformedTri.edge2 = transformedTri.v2 - transformedTri.v0;
                transformedTri.deltaUV1 = transformedTri.uv1 - transformedTri.uv0;
                transformedTri.deltaUV2 = transformedTri.uv2 - transformedTri.uv0;

                transformedTri.materialId = newGeom.materialid;

                //triangles[i] = transformedTri;
                triangles.push_back(transformedTri);

                //geoms.push_back(newGeom);




            }
            std::cout << "Total triangles after this mesh: " << triangles.size() << "\n";


            //for (int i = 0; i < triangles.size(); i++) {
            //    std::cout << "Triangle " << i << ": materialId=" << triangles[i].materialId
            //        << " v0=(" << triangles[i].v0.x << "," << triangles[i].v0.y << "," << triangles[i].v0.z << ")\n";
            //}

            //if (!triangles.empty()) {
            //    std::cout << "Building BVH for all " << triangles.size() << " triangles\n";
            //    int N = triangles.size();
            //    bvhNodes.clear();
            //    bvhNodes.resize(N * 2 - 1);
            //    BuildBVH(bvhNodes, N);
            //}
            //std::cout << "[BVH] Built with " << nodesUsed << " nodes for " << N << " triangles\n";
            //std::cout << "[BVH] triIdx size: " << triIdx.size() << "\n";

            //std::cout << "[BVH] Root AABB: min=(" << bvhNodes[0].aabbMin.x << ","
            //    << bvhNodes[0].aabbMin.y << "," << bvhNodes[0].aabbMin.z << ") max=("
            //    << bvhNodes[0].aabbMax.x << "," << bvhNodes[0].aabbMax.y << ","
            //    << bvhNodes[0].aabbMax.z << ")\n";

            //std::cout << "[GLTF-TRIS] " << triangles.size() << "\n";

        }
        else {
            Geom newGeom;
            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else if (type == "sphere")
            {
                newGeom.type = SPHERE;
            }
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
            geoms.push_back(newGeom);
        }

    }

    // BUILD BVH ONCE after ALL meshes loaded
    if (!triangles.empty()) {
        //std::cout << "\nBuilding BVH for all " << triangles.size() << " triangles\n";
        int N = triangles.size();
        bvhNodes.clear();
        bvhNodes.resize(N * 2 - 1);
        triIdx.clear();  // Important: clear triIdx before building
        rootNodeIdx = 0;
        nodesUsed = 1;
        BuildBVH(bvhNodes, N);
        //std::cout << "BVH complete with " << nodesUsed << " nodes\n";
    }

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    if (cameraData.contains("LENS_RADIUS")) {
        camera.lensRadius = cameraData["LENS_RADIUS"];
    }
    else {
        camera.lensRadius = 0.0f;
    }

    if (cameraData.contains("FOCAL_DISTANCE")) {
        camera.focalDistance = cameraData["FOCAL_DISTANCE"];
    }
    else {
        camera.focalDistance = 10.0f;
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());


}



//BVHNode bvhNode[N * 2 - 1];
//int rootNodeIdx = 0, nodesUsed = 1;


void Scene::BuildBVH(std::vector<BVHNode>& bvhNodes, int N)
{

    for (int i = 0; i < N; ++i) {
        triangles[i].centroid = (triangles[i].v0 + triangles[i].v1 + triangles[i].v2) * 0.3333f;
        triIdx.push_back(i);
    }

    BVHNode& root = bvhNodes[rootNodeIdx];
    root.firstPrim = 0;
    root.primCount = N;

    UpdateNodeBounds(rootNodeIdx, bvhNodes, N);
    Subdivide(rootNodeIdx, bvhNodes, N);


}
void Scene::UpdateNodeBounds(int nodeIdx, std::vector<BVHNode>& bvhNodes, int N)
{
    BVHNode& node = bvhNodes[nodeIdx];
    node.aabbMin = glm::vec3(1e30f);
    node.aabbMax = glm::vec3(-1e30f);

    for (int i = 0; i < node.primCount; i++)
    {
        int triIndex = triIdx[node.firstPrim + i];
        MeshTriangle& leafTri = triangles[triIndex];

        node.aabbMin = glm::min(node.aabbMin, leafTri.v0);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v1);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v2);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v0);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v1);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v2);
    }

    // Expand by epsilon to handle flat triangles
    glm::vec3 epsilon(0.001f);
    node.aabbMin -= epsilon;
    node.aabbMax += epsilon;
}

void Scene::Subdivide(int nodeIdx, std::vector<BVHNode>& bvhNodes, int N)
{
    // terminate recursion
    BVHNode& node = bvhNodes[nodeIdx];
    if (node.primCount <= 2) return;
    // determine split axis and position
    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
    // in-place partition
    int i = node.firstPrim;
    int j = i + node.primCount - 1;
    while (i <= j)
    {
        if (triangles[triIdx[i]].centroid[axis] < splitPos)
            i++;
        else
            swap(triIdx[i], triIdx[j--]);
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.firstPrim;
    if (leftCount == 0 || leftCount == node.primCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    bvhNodes[leftChildIdx].firstPrim = node.firstPrim;
    bvhNodes[leftChildIdx].primCount = leftCount;
    bvhNodes[rightChildIdx].firstPrim = i;
    bvhNodes[rightChildIdx].primCount = node.primCount - leftCount;
    node.leftChild = leftChildIdx;
    node.rightChild = rightChildIdx;
    node.primCount = 0;
    UpdateNodeBounds(leftChildIdx, bvhNodes, N);
    UpdateNodeBounds(rightChildIdx, bvhNodes, N);
    // recurse
    Subdivide(leftChildIdx, bvhNodes, N);
    Subdivide(rightChildIdx, bvhNodes, N);
}
