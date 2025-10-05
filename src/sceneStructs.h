#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRI
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int triangle_index;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    int textureID;
    int normalMapID;
    bool hasTexture;
    bool hasNormalMap;

    // ADD PBR PROPERTIES
    float metallic;
    float roughness;
    int metallicRoughnessTextureID;
    bool hasMetallicRoughnessTexture;
};

struct Texture {
    int width;
    int height;
    glm::vec3* data;

};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
  int triangleIndex;
};



struct BVHNode {
    // leaf if primCount > 0
    int firstPrim = 0;
    int primCount = 0;
    // children if inner node
    int leftChild = -1;
    int rightChild = -1;
    // world-space AABB
    glm::vec3 aabbMin = glm::vec3(1e30f);
    glm::vec3 aabbMax = glm::vec3(-1e30f);
    bool isLeaf() { return primCount > 0; };
};
