#include "intersections.h"
#include "sceneStructs.h"
#include "gltfLoader.h"
#include "utilities.h"
//#include <wincrypt.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

__host__ __device__ float triangleIntersectionTest(
    Geom triangle,
    Ray r,
    const MeshTriangle& tri,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // Moller-Trumbore intersection algorithm
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;
    glm::vec3 h = glm::cross(r.direction, edge2);
    float a = glm::dot(edge1, h);

    // Parallel to triangle plane
    if (a > -EPSILON && a < EPSILON)
        return -1.0f;

    float f = 1.0f / a;
    glm::vec3 s = r.origin - tri.v0;
    float u = f * glm::dot(s, h);

    // Check barycentric coordinate u
    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(r.direction, q);

    // Check barycentric coordinate v
    if (v < 0.0f || u + v > 1.0f)
        return -1.0f;

    float t = f * glm::dot(edge2, q);

    if (t > EPSILON) {
        intersectionPoint = getPointOnRay(r, t);
        normal = glm::normalize(glm::cross(edge1, edge2));

        // Double-sided rendering: flip normal to face ray if needed
        if (glm::dot(normal, r.direction) > 0.0f) {
            normal = -normal;
        }

        outside = true;
        return t;
    }

    return -1.0f;
}



__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

// BEFORE this function, make sure you have:  #include <cfloat>
__host__ __device__ inline bool solveQuadratic(float A, float B, float C, float& t0, float& t1) {
    // Normalize to monic: t^2 + b t + c = 0
    const float invA = 1.0f / A;
    const float b = B * invA;
    const float c = C * invA;

    const float half_b = 0.5f * b;
    const float disc = half_b * half_b - c;
    if (disc < 0.0f) { t0 = t1 = FLT_MAX; return false; }

    const float s = sqrtf(disc);

    const float q = (half_b > 0.0f) ? -(half_b + s) : -(half_b - s);
    t0 = q;
    t1 = c / q;

    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
    return true;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    /*
    float t0, t1;
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f));


    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float a = glm::dot(rt.direction, rt.direction);
    glm::vec3 diff = rt.origin;
   
    float b = 2.0 * glm::dot(rt.direction, diff);
    float c = glm::dot(diff, diff) - (radius * radius);

    if (!solveQuadratic(a, b, c, t0, t1)) {
        return -1.0f;
    }


    float t_obj = (t0 > 0.0f) ? t0 : ((t1 > 0.0f) ? t1 : -1.0f);
    if (t_obj < 0.0f) return -1.0f;

    const glm::vec3 p_obj = ro + t_obj * rd;

    glm::vec3 n_obj = p_obj / radius; 
    n_obj = glm::normalize(n_obj);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(p_obj, 1.0f));
    normal = multiplyMV(sphere.invTranspose, glm::vec4(n_obj, 0.0f));
    normal = glm::normalize(normal);

    outside = (t0 > 0.0f);

    const float worldT = glm::length(intersectionPoint - r.origin);
    return worldT;
    */

    
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = fmin(t1, t2);
        outside = true;
    }
    else
    {
        t = fmax(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    //if (!outside)
    //{
    //    normal = -normal;
    //}

    return glm::length(r.origin - intersectionPoint);
    
}



__host__ __device__ float triangleIntersectionTestRaw(
    Ray r,
    const MeshTriangle& tri,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv)
{
    // Moller-Trumbore intersection algorithm
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;
    glm::vec3 h = glm::cross(r.direction, edge2);
    float a = glm::dot(edge1, h);

    if (a > -0.00001f && a < 0.00001f) return -1.0f;

    float f = 1.0f / a;
    glm::vec3 s = r.origin - tri.v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) return -1.0f;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(r.direction, q);

    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    float t = f * glm::dot(edge2, q);

    if (t > 0.00001f) {
        intersectionPoint = r.origin + t * r.direction;
        normal = glm::normalize(glm::cross(edge1, edge2));
        outside = glm::dot(r.direction, normal) < 0.0f;

        float w = 1.0f - u - v;
        uv = w * tri.uv0 + u * tri.uv1 + v * tri.uv2;

        return t;
    }

    return -1.0f;
}

__host__ __device__ float IntersectBVH(
    Ray r,
    MeshTriangle* triangles,
    const int rootNodeIdx,
    BVHNode* bvhNodes,
    int* triIdx,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    int& hitTriIdx,
    glm::vec2& uv)
{
    if (!bvhNodes || !triangles || !triIdx) return -1.0f;

    int stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = rootNodeIdx;

    float tMin = FLT_MAX;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        BVHNode& node = bvhNodes[nodeIdx];

        bool aabbHit = IntersectAABB(r, node.aabbMin, node.aabbMax);

#ifndef __CUDA_ARCH__
        if (nodeIdx == 0) {
            printf("AABB test: hit=%d, ray.origin=(%.2f,%.2f,%.2f), ray.dir=(%.2f,%.2f,%.2f)\n",
                aabbHit, r.origin.x, r.origin.y, r.origin.z,
                r.direction.x, r.direction.y, r.direction.z);
        }
#endif

        if (!aabbHit) continue;

        if (node.primCount > 0) {  // Leaf
            for (int i = 0; i < node.primCount; i++) {
                int triIndex = triIdx[node.firstPrim + i];
                glm::vec3 tmpInt, tmpNorm;
                bool tmpOut;
                glm::vec2 tmpUV;

                float t = triangleIntersectionTestRaw(r, triangles[triIndex],
                    tmpInt, tmpNorm, tmpOut, tmpUV);

#ifndef __CUDA_ARCH__
                printf("Triangle test: triIdx=%d, t=%f\n", triIndex, t);
#endif

                if (t > 0.0f && t < tMin) {
                    tMin = t;
                    intersectionPoint = tmpInt;
                    normal = tmpNorm;
                    outside = tmpOut;
                    hitTriIdx = triIndex;
                    uv = tmpUV;
                }
            }
        }
        else {
            if (node.rightChild >= 0) stack[stackPtr++] = node.rightChild;
            if (node.leftChild >= 0) stack[stackPtr++] = node.leftChild;
        }
    }

    return tMin < FLT_MAX ? tMin : -1.0f;
}


// intersections.h  (CHANGE: provide inline impl)
__host__ __device__ inline bool IntersectAABB(const Ray& ray,
    const glm::vec3 bmin,
    const glm::vec3 bmax)
{
    // slab test; no reliance on ray.t
    const glm::vec3 invD = 1.0f / ray.direction;

    const glm::vec3 t0 = (bmin - ray.origin) * invD;
    const glm::vec3 t1 = (bmax - ray.origin) * invD;

    const glm::vec3 tmin3 = glm::min(t0, t1);
    const glm::vec3 tmax3 = glm::max(t0, t1);

    const float tNear = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
    const float tFar = fminf(fminf(tmax3.x, tmax3.y), tmax3.z);

    // hit if non-empty overlap and any part is in front
    return (tFar >= 0.0f) && (tNear <= tFar);
}