# GPU-Accelerated Image Processing using CUDA

```
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

struct Pixel {
    unsigned char r, g, b;
};

__global__ void grayscaleKernel(Pixel *input, Pixel *output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (idx < totalPixels) {
        unsigned char gray = 0.299f * input[idx].r +
                             0.587f * input[idx].g +
                             0.114f * input[idx].b;

        output[idx].r = gray;
        output[idx].g = gray;
        output[idx].b = gray;
    }
}

__global__ void brightnessKernel(Pixel *input, Pixel *output, int width, int height, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (idx < totalPixels) {
        int r = input[idx].r + value;
        int g = input[idx].g + value;
        int b = input[idx].b + value;

        output[idx].r = min(max(r, 0), 255);
        output[idx].g = min(max(g, 0), 255);
        output[idx].b = min(max(b, 0), 255);
    }
}

bool readPPM(const string &filename, vector<Pixel> &image, int &width, int &height) {
    ifstream file(filename, ios::binary);

    if (!file) {
        cout << "Error: Cannot open input image.\n";
        return false;
    }

    string format;
    int maxColor;

    file >> format;
    file >> width >> height;
    file >> maxColor;
    file.ignore();

    if (format != "P6") {
        cout << "Error: Only P6 PPM format is supported.\n";
        return false;
    }

    image.resize(width * height);
    file.read(reinterpret_cast<char *>(image.data()), width * height * sizeof(Pixel));

    return true;
}

bool writePPM(const string &filename, const vector<Pixel> &image, int width, int height) {
    ofstream file(filename, ios::binary);

    if (!file) {
        cout << "Error: Cannot create output image.\n";
        return false;
    }

    file << "P6\n";
    file << width << " " << height << "\n";
    file << "255\n";

    file.write(reinterpret_cast<const char *>(image.data()), width * height * sizeof(Pixel));

    return true;
}

int main() {
    string inputFile = "input.ppm";
    string grayOutputFile = "output_grayscale.ppm";
    string brightOutputFile = "output_brightness.ppm";

    int width, height;
    vector<Pixel> hostInput;

    if (!readPPM(inputFile, hostInput, width, height)) {
        return 1;
    }

    int totalPixels = width * height;
    size_t imageSize = totalPixels * sizeof(Pixel);

    vector<Pixel> hostGrayOutput(totalPixels);
    vector<Pixel> hostBrightOutput(totalPixels);

    Pixel *deviceInput, *deviceGrayOutput, *deviceBrightOutput;

    cudaMalloc(&deviceInput, imageSize);
    cudaMalloc(&deviceGrayOutput, imageSize);
    cudaMalloc(&deviceBrightOutput, imageSize);

    cudaMemcpy(deviceInput, hostInput.data(), imageSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    grayscaleKernel<<<blocks, threadsPerBlock>>>(deviceInput, deviceGrayOutput, width, height);
    brightnessKernel<<<blocks, threadsPerBlock>>>(deviceInput, deviceBrightOutput, width, height, 50);

    cudaMemcpy(hostGrayOutput.data(), deviceGrayOutput, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostBrightOutput.data(), deviceBrightOutput, imageSize, cudaMemcpyDeviceToHost);

    writePPM(grayOutputFile, hostGrayOutput, width, height);
    writePPM(brightOutputFile, hostBrightOutput, width, height);

    cudaFree(deviceInput);
    cudaFree(deviceGrayOutput);
    cudaFree(deviceBrightOutput);

    cout << "GPU image processing completed successfully.\n";
    cout << "Grayscale image saved as: " << grayOutputFile << endl;
    cout << "Brightness image saved as: " << brightOutputFile << endl;

    return 0;
}
```
