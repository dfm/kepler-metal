#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <chrono>
#include <cmath>
#include <iostream>

int main(int argc, const char *argv[]) {
  NS::Error *error;
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  // Set up the device
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    std::cerr << "Failed to find device.\n";
    exit(-1);
  }

  auto lib = device->newDefaultLibrary();
  if (!lib) {
    std::cerr << "Failed to find the default library.\n";
    exit(-1);
  }

  auto fn_name = NS::String::string("solve_kepler", NS::ASCIIStringEncoding);
  auto computeFunction = lib->newFunction(fn_name);
  if (!computeFunction) {
    std::cerr << "Failed to find the compute function.\n";
    exit(-1);
  }

  auto pipeline_state = device->newComputePipelineState(computeFunction, &error);
  if (!pipeline_state) {
    std::cerr << "Failed to create the pipeline state object.\n";
    exit(-1);
  }

  auto cmd_queue = device->newCommandQueue();
  if (!cmd_queue) {
    std::cerr << "Failed to find command queue.\n";
    exit(-1);
  }

  const auto ARRAY_SIZE = 1024 * 1024;
  const auto BUFFER_SIZE = ARRAY_SIZE * sizeof(float);
  auto buffer1 = device->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  auto buffer2 = device->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  auto buffer3 = device->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
  for (ssize_t i = 0; i < ARRAY_SIZE; i++) {
    ((float *)buffer1->contents())[i] = 2 * M_PI * i / (ARRAY_SIZE + 1.0);
    ((float *)buffer2->contents())[i] = 2 * M_PI * i / (ARRAY_SIZE + 1.0);
  }

  auto start = std::chrono::steady_clock::now();
  // Actually do the computation
  MTL::CommandBuffer *cmd_buf = cmd_queue->commandBuffer();
  if (!cmd_buf) {
    std::cerr << "Failed to create command buffer.\n";
    exit(-1);
  }

  // Start a compute pass.
  auto *cmd_enc = cmd_buf->computeCommandEncoder();
  if (!cmd_enc) {
    std::cerr << "Failed to create command encoder.\n";
    exit(-1);
  }

  // Encode the pipeline state object and its parameters.
  cmd_enc->setComputePipelineState(pipeline_state);
  cmd_enc->setBuffer(buffer1, 0, 0);
  cmd_enc->setBuffer(buffer2, 0, 1);
  cmd_enc->setBuffer(buffer3, 0, 2);

  // Calculate the problem dimensions.
  auto size = pipeline_state->maxTotalThreadsPerThreadgroup();
  if (size > ARRAY_SIZE) {
    size = ARRAY_SIZE;
  }
  auto grid_size = MTL::Size(ARRAY_SIZE, 1, 1);
  auto group_size = MTL::Size(size, 1, 1);

  cmd_enc->dispatchThreads(grid_size, group_size);
  cmd_enc->endEncoding();

  cmd_buf->commit();              // Actually execute the command
  cmd_buf->waitUntilCompleted();  // Block until ready

  auto end = std::chrono::steady_clock::now();
  auto delta_time = end - start;

  std::cout << "Computation completed in "
            << std::chrono::duration<double, std::milli>(delta_time).count()
            << " ms for array of size " << ARRAY_SIZE << ".\n";

  device->release();
  pool->release();
  return 0;
}
