#pragma once

#include "scene.h"
#include "utilities.h"

#define doDenoise 1
#define doRussianRoulette 1

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void denoise(glm::vec3* colorBuffer, glm::vec3* normalBuffer, int width, int height);