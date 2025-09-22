#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


__host__ __device__ glm::vec3 squareToDiskConcentric(const glm::vec2& xi) {
    float a = 2.0f * xi.x - 1.0f;
    float b = 2.0f * xi.y - 1.0f;

    float r, phi;
    if (a > -b) {
        if (a > b) { r = a; phi = (PI * 0.25f) * (b / a); }
        else { r = b; phi = (PI * 0.25f) * (2.0f - (a / b)); }
    }
    else {
        if (a < b) { r = -a; phi = (PI * 0.25f) * (4.0f + (b / a)); }
        else { r = -b; phi = (b != 0.0f) ? (PI * 0.25f) * (6.0f - (a / b)) : 0.0f; }
    }

    float u = r * cosf(phi);
    float v = r * sinf(phi);
    return glm::vec3(u, v, 0.0f);
}

// Map square -> cosine-weighted hemisphere (local +Z is the normal)
__host__ __device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi) {
    glm::vec3 d = squareToDiskConcentric(xi);
    float x = d.x, y = d.y;
    float z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));
    return glm::normalize(glm::vec3(x, y, z));
}

// Build an ONB for a given normal
__host__ __device__ inline void buildOnb(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    // choose a helper vector not parallel to n
    glm::vec3 h = (fabsf(n.x) > 0.1f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    b = glm::normalize(glm::cross(n, h));
    t = glm::cross(b, n);
}

// World-space cosine hemisphere sample w/ matching pdf
__host__ __device__ glm::vec3 cosineSampleHemisphere(const glm::vec3& n,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    glm::vec2 xi(u01(rng), u01(rng));            // two uniform randoms in [0,1)
    glm::vec3 local = squareToHemisphereCosine(xi);

    // rotate local (+Z up) into world frame defined by n
    glm::vec3 t, b; buildOnb(n, t, b);
    glm::vec3 wi = glm::normalize(local.x * t + local.y * b + local.z * n);

    float cosTheta = fmaxf(0.f, glm::dot(n, wi));
    return wi;
}

__host__ __device__ void bsdf_pdf(PathSegment& pathSegment,
    glm::vec3 normal, float& pdf)
{
    float cosTheta = glm::max(0.f, glm::dot(pathSegment.ray.direction, normal));
    pdf = cosTheta / PI;
}


__host__ __device__ void bsdf_specular(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    float& pdf,
    const Material& m,
    thrust::default_random_engine& rng)
{

    glm::vec3 dir = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect + EPSILON * normal;
    //pathSegment.ray.direction = dir;
    pathSegment.ray.direction = dir;
    float cosTheta = glm::max(0.f, glm::dot(pathSegment.ray.direction, normal));
    bsdf_pdf(pathSegment, normal, pdf);
    pathSegment.color *= m.color / cosTheta;
}


__host__ __device__ void bsdf_diffuse(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    float& pdf,
    const Material& m,
    thrust::default_random_engine& rng)
{

    glm::vec3 dir = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect + EPSILON * normal;
    //pathSegment.ray.direction = dir;
    pathSegment.ray.direction = dir;

    bsdf_pdf(pathSegment, normal, pdf);
    pathSegment.color *= m.color;
}


__host__ __device__ void bsdf(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    float& pdf,
    const Material& m,
    thrust::default_random_engine& rng)
{
    if (m.hasReflective) {
        bsdf_specular(pathSegment, intersect, normal, pdf, m, rng);

    }
    else {
        bsdf_diffuse(pathSegment, intersect, normal, pdf, m, rng);
    }

}



__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    /*
    glm::vec3 dir = calculateRandomDirectionInHemisphere(normal, rng);

    pathSegment.ray.origin = intersect + EPSILON * normal;
    //pathSegment.ray.direction = dir;
    pathSegment.ray.direction = glm::normalize(dir);


    pathSegment.color *= m.color;
    */
    float pdf;
    bsdf(pathSegment, intersect, normal, pdf, m, rng);

}
