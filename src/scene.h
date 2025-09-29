#pragma once

#include "sceneStructs.h"
#include <vector>
#include "glTFLoader.h"
#include <unordered_map>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    bool jsonLoadedNonCuda = false;
    std::string jsonName_str;
    int triangleCount = -1;
    std::vector<MeshTriangle> triangles;
public:
    Scene(std::string filename);

    std::vector<MeshTriangle> getTriangleBuffer();
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
