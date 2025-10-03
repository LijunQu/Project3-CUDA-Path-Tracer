#pragma once

#include "sceneStructs.h"
#include <vector>
#include "glTFLoader.h"
#include <unordered_map>
#include "stb_image.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    bool jsonLoadedNonCuda = false;
    std::string jsonName_str;
    int triangleCount = -1;


    void BuildBVH(std::vector<BVHNode>& bvhNodes, int N);
    void UpdateNodeBounds(int nodeIdx, std::vector<BVHNode>& bvhNodes, int N); 
    void Subdivide(int nodeIdx, std::vector<BVHNode>& bvhNodes, int N);
public:
    Scene(std::string filename);
    std::vector<Texture> textures;
    Texture loadTexture(const std::string& filepath);

    std::vector<MeshTriangle> getTriangleBuffer();
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<MeshTriangle> triangles;
    std::vector<int> triIdx;
    std::vector<BVHNode> bvhNodes;

    int rootNodeIdx = 0, nodesUsed = 1;
};
