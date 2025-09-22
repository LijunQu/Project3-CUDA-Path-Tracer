#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ glm::vec3 squareToDiskConcentric(const glm::vec2& xi);
__host__ __device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi);

// Cosine-weighted sample in WORLD space given a shading normal `n`.
// Returns direction; writes matching pdf = cos(theta)/PI.
__host__ __device__ glm::vec3 cosineSampleHemisphere(const glm::vec3& n,
                                            thrust::default_random_engine& rng,
                                            float& outPdf);

__host__ __device__ void bsdf_pdf(PathSegment& pathSegment,
    glm::vec3 normal, float& pdf);

__host__ __device__ void bsdf_specular(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    float& pdf,
    const Material& m,
    thrust::default_random_engine& rng);

__host__ __device__ void bsdf_diffuse(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    float& pdf,
    const Material& m,
    thrust::default_random_engine& rng);

__host__ __device__ void bsdf(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    float& pdf,
    const Material& m,
    thrust::default_random_engine& rng);

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);
