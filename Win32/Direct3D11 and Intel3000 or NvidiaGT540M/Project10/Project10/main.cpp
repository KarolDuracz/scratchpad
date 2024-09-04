#if 0
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h> // Include the D3DCompiler header
#include <iostream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "D3DCompiler.lib") // Link against the D3DCompiler library

const char* computeShaderSource = R"(
    RWStructuredBuffer<float> Result : register(u0);

    [numthreads(1, 1, 1)]
    void main(uint3 DTid : SV_DispatchThreadID)
    {
        // Simple computation: square the index
        Result[DTid.x] = DTid.x * DTid.x;
    }
)";

ID3D11ComputeShader* createComputeShader(ID3D11Device* device, const char* shaderSource) {
    ID3D11ComputeShader* computeShader = nullptr;

    ID3DBlob* csBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;

    HRESULT hr = D3DCompile(shaderSource, strlen(shaderSource), nullptr, nullptr, nullptr,
        "main", "cs_5_0", 0, 0, &csBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Compute shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        if (csBlob) csBlob->Release();
        return nullptr;
    }

    hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &computeShader);
    csBlob->Release();

    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader." << std::endl;
        return nullptr;
    }

    return computeShader;
}

int main() {
    IDXGIFactory* pFactory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);

    if (FAILED(hr)) {
        std::cerr << "Failed to create DXGIFactory." << std::endl;
        return -1;
    }

    IDXGIAdapter* pAdapter = nullptr;

    // Attempt to get the second GPU (index 1)
    hr = pFactory->EnumAdapters(1, &pAdapter);  // Index 1 for the second GPU
    if (hr == DXGI_ERROR_NOT_FOUND) {
        std::cerr << "Second GPU not found." << std::endl;
        pFactory->Release();
        return -1;
    }

    DXGI_ADAPTER_DESC adapterDesc;
    pAdapter->GetDesc(&adapterDesc);

    // Output basic information about the second adapter (GPU)
    std::wcout << L"Using Adapter: " << adapterDesc.Description << std::endl;
    std::wcout << L"Vendor ID: " << adapterDesc.VendorId << std::endl;
    std::wcout << L"Device ID: " << adapterDesc.DeviceId << std::endl;

    // Attempt to create a Direct3D 11 device on the second GPU
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL featureLevel;
    hr = D3D11CreateDevice(
        pAdapter,                // Use the second adapter (GPU)
        D3D_DRIVER_TYPE_UNKNOWN, // Must be UNKNOWN when specifying an adapter
        nullptr,                 // No software rasterizer
        0,                       // Flags (e.g., debug layer)
        nullptr,                 // Feature levels (null for all)
        0,                       // Number of feature levels
        D3D11_SDK_VERSION,       // SDK version
        &device,                 // Device output
        &featureLevel,           // Feature level output
        &context                 // Device context output
    );

    if (SUCCEEDED(hr)) {
        std::wcout << L"Direct3D 11 is supported on this GPU." << std::endl;

        // Creating the compute shader
        ID3D11ComputeShader* computeShader = createComputeShader(device, computeShaderSource);
        if (!computeShader) {
            context->Release();
            device->Release();
            pAdapter->Release();
            pFactory->Release();
            return -1;
        }

        // Creating a buffer to store the results
        const int numElements = 1000;
        D3D11_BUFFER_DESC bufferDesc = {};
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.ByteWidth = sizeof(float) * numElements;
        bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        bufferDesc.CPUAccessFlags = 0; // No CPU access

        ID3D11Buffer* resultBuffer = nullptr;
        hr = device->CreateBuffer(&bufferDesc, nullptr, &resultBuffer);
        if (FAILED(hr)) {
            std::cerr << "Failed to create result buffer. HRESULT: " << std::hex << hr << std::endl;
            computeShader->Release();
            context->Release();
            device->Release();
            pAdapter->Release();
            pFactory->Release();
            return -1;
        }

        // Creating unordered access view for the buffer
        ID3D11UnorderedAccessView* uav = nullptr;
        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = numElements;

        hr = device->CreateUnorderedAccessView(resultBuffer, &uavDesc, &uav);
        if (FAILED(hr)) {
            std::cerr << "Failed to create UAV. HRESULT: " << std::hex << hr << std::endl;
            resultBuffer->Release();
            computeShader->Release();
            context->Release();
            device->Release();
            pAdapter->Release();
            pFactory->Release();
            return -1;
        }

        // Binding the compute shader and UAV
        context->CSSetShader(computeShader, nullptr, 0);
        context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);

        // Dispatching the compute shader
        context->Dispatch(numElements, 1, 1);

        // Unbind and clean up
        context->CSSetShader(nullptr, nullptr, 0);
        ID3D11UnorderedAccessView* nullUAV = nullptr;
        context->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);

        // Clean up
        uav->Release();
        resultBuffer->Release();
        computeShader->Release();
    }
    else {
        std::wcout << L"Direct3D 11 is not supported on this GPU." << std::endl;
    }

    context->Release();
    device->Release();
    pAdapter->Release();
    pFactory->Release();

    return 0;
}
#endif


#if 1
#include <d3d11.h>
#include <dxgi.h>
#include <iostream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

void Benchmark(ID3D11Device* device) {
    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);

    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;

    ID3D11Query* query = nullptr;
    device->CreateQuery(&queryDesc, &query);

    context->Begin(query);

    // Create an offscreen render target (dummy texture)
    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = 1;  // Minimal size
    texDesc.Height = 1; // Minimal size
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_RENDER_TARGET;

    ID3D11Texture2D* renderTargetTexture = nullptr;
    device->CreateTexture2D(&texDesc, nullptr, &renderTargetTexture);

    // Create a render target view
    ID3D11RenderTargetView* renderTargetView = nullptr;
    device->CreateRenderTargetView(renderTargetTexture, nullptr, &renderTargetView);

    // Bind the render target view
    context->OMSetRenderTargets(1, &renderTargetView, nullptr);

    // Minimal vertex buffer setup for a single triangle (data is not used)
    ID3D11Buffer* vertexBuffer = nullptr;
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = 3 * sizeof(float);  // Just enough for one vertex position
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    device->CreateBuffer(&bufferDesc, nullptr, &vertexBuffer);

    // Simple rendering loop
    for (int i = 0; i < 100; ++i) {
        // Here you would put rendering code, e.g., drawing a triangle
         // Clear the render target (not actually drawing anything useful)
        float clearColor[4] = { 0, 0, 0, 0 };
        context->ClearRenderTargetView(renderTargetView, clearColor);

        // Perform a dummy draw call
        UINT stride = sizeof(float) * 3;
        UINT offset = 0;
        context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->Draw(3, 0);
    }

    context->End(query);

    while (context->GetData(query, nullptr, 0, 0) == S_FALSE) {}

    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjointData;
    context->GetData(query, &disjointData, sizeof(disjointData), 0);

    if (!disjointData.Disjoint) {
        std::cout << "Benchmark completed successfully" << std::endl;
    }

    query->Release();
    context->Release();
}

void MeasureGPUUtilization(ID3D11Device* device) {
    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);

    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_PIPELINE_STATISTICS;

    ID3D11Query* query = nullptr;
    device->CreateQuery(&queryDesc, &query);

    context->Begin(query);

    // Perform rendering or computation

    context->End(query);

    D3D11_QUERY_DATA_PIPELINE_STATISTICS stats;
    while (context->GetData(query, &stats, sizeof(stats), 0) == S_FALSE) {}

    std::cout << "VertexShader Invocations: " << stats.VSInvocations << std::endl;
    std::cout << "PixelShader Invocations: " << stats.PSInvocations << std::endl;

    query->Release();
    context->Release();
}

int main() {
    IDXGIFactory* pFactory = nullptr;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);

    IDXGIAdapter* pAdapter = nullptr;
    for (UINT i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC adapterDesc;
        pAdapter->GetDesc(&adapterDesc);

        std::wcout << L"GPU " << i << L": " << adapterDesc.Description << std::endl;

        // Create Direct3D device
        ID3D11Device* device = nullptr;
        D3D_FEATURE_LEVEL featureLevel;
        D3D11CreateDevice(pAdapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &featureLevel, nullptr);

        Benchmark(device);
        MeasureGPUUtilization(device);

        device->Release();
        pAdapter->Release();
    }

    pFactory->Release();
    return 0;
}
#endif