#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <vector>
#include <string>
#include <iostream>

// Link the necessary libraries
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

// Global variables
ID3D11Device* gDevice = nullptr;
ID3D11DeviceContext* gDeviceContext = nullptr;
ID3D11RenderTargetView* gRenderTargetView = nullptr;
std::vector<ID3D11ShaderResourceView*> gTextureViews;
const int numFrames = 10; // Update with the actual number of frames

// Function prototypes
HRESULT LoadTextureFromFile(const wchar_t* fileName, ID3D11ShaderResourceView** textureView);
void LoadAllTextures();
void RenderFrame(int frameIndex);
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

// Win32 window setup
HWND CreateMyWindow(HINSTANCE hInstance) {
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"Direct3D11WindowClass";

    RegisterClass(&wc);

    return CreateWindowEx(NULL, wc.lpszClassName, L"Direct3D 11 Video Renderer", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, nullptr, nullptr, hInstance, nullptr);
}

void InitD3D(HWND hwnd) {

    // Create the Direct3D device and swap chain
    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferCount = 1;
    swapChainDesc.BufferDesc.Width = 800;
    swapChainDesc.BufferDesc.Height = 600;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = hwnd;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.Windowed = TRUE;

    // Read the texture data and create a D3D11 texture
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = 800;
    textureDesc.Height = 600;
    textureDesc.MipLevels = 4;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // or appropriate format based on header
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_IMMUTABLE;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION, &swapChainDesc, nullptr, &gDevice, nullptr, &gDeviceContext);

    // Create render target view
    ID3D11Texture2D* backBuffer = nullptr;
    gDevice->CreateTexture2D(&textureDesc, nullptr, &backBuffer);
    gDevice->CreateRenderTargetView(backBuffer, nullptr, &gRenderTargetView);
    backBuffer->Release();
}

void LoadAllTextures() {
    for (int i = 1; i <= numFrames; ++i) {
        std::wstring frameFilename = L"C:\\Windows\\Temp\\v2\\video_frame_";
        frameFilename += std::to_wstring(i);
        frameFilename += L".dds";

        ID3D11ShaderResourceView* textureView = nullptr;
        HRESULT hr = LoadTextureFromFile(frameFilename.c_str(), &textureView);
        if (SUCCEEDED(hr)) {
            gTextureViews.push_back(textureView);
        }
        else {
            std::wcerr << L"Failed to load texture: " << frameFilename << std::endl;
        }
    }
}

void RenderFrame(int frameIndex) {
    gDeviceContext->OMSetRenderTargets(1, &gRenderTargetView, nullptr);
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    gDeviceContext->ClearRenderTargetView(gRenderTargetView, clearColor);

    // Bind the current frame texture to the pixel shader
    gDeviceContext->PSSetShaderResources(0, 1, &gTextureViews[frameIndex]);

    // Draw your geometry here (not shown for brevity)

    // Swap buffers, etc. (not shown for brevity)
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    HWND hwnd = CreateMyWindow(hInstance);
    ShowWindow(hwnd, SW_SHOW);
    InitD3D(hwnd);

    LoadAllTextures();

    // Main loop
    int currentFrame = 0;
    MSG msg = {};
    while (msg.message != WM_QUIT) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Render the current frame
        RenderFrame(currentFrame);
        currentFrame = (currentFrame + 1) % gTextureViews.size(); // Loop through frames

        // Present the rendered frame (swap buffers, etc.)
    }

    // Cleanup (release resources)
    for (auto texture : gTextureViews) {
        texture->Release();
    }
    gDeviceContext->Release();
    gDevice->Release();
    gRenderTargetView->Release();

    return 0;
}

HRESULT LoadTextureFromFile(const wchar_t* fileName, ID3D11ShaderResourceView** textureView) {
    // Implement loading DDS file here
    // Use dds.c or any DDS loading utility you have
    // For now, here’s a stub returning S_OK
    return S_OK;
}




// cos jest ale to nie dziala bo trzeba ladowac kazda frame jakos
#if 0
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <d3d11.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>

#include <stdio.h>
#include <wchar.h>


#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

struct DDS_PIXELFORMAT {
    DWORD dwSize;
    DWORD dwFlags;
    DWORD dwFourCC;
    DWORD dwRGBBitCount;
    DWORD dwRBitMask;
    DWORD dwGBitMask;
    DWORD dwBBitMask;
    DWORD dwABitMask;
};

typedef struct {
    DWORD           dwSize;
    DWORD           dwFlags;
    DWORD           dwHeight;
    DWORD           dwWidth;
    DWORD           dwPitchOrLinearSize;
    DWORD           dwDepth;
    DWORD           dwMipMapCount;
    DWORD           dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    DWORD           dwCaps;
    DWORD           dwCaps2;
    DWORD           dwCaps3;
    DWORD           dwCaps4;
    DWORD           dwReserved2;
} DDS_HEADER;

using namespace DirectX;

struct Vertex {
    XMFLOAT3 position;
    XMFLOAT2 texcoord;
};

// Global D3D variables
IDXGISwapChain* gSwapChain = nullptr;
ID3D11Device* gDevice = nullptr;
ID3D11DeviceContext* gDeviceContext = nullptr;
ID3D11RenderTargetView* gRenderTargetView = nullptr;
ID3D11Buffer* gVertexBuffer = nullptr;
ID3D11Buffer* gConstantBuffer = nullptr;
ID3D11VertexShader* gVertexShader = nullptr;
ID3D11PixelShader* gPixelShader = nullptr;
ID3D11InputLayout* gInputLayout = nullptr;
ID3D11ShaderResourceView* gTextureView = nullptr;
ID3D11SamplerState* gSamplerState = nullptr;

ID3D11Texture2D* gTexture = nullptr;

float gRotationAngle = 0.0f;
HWND gHwnd = nullptr;

Vertex vertices[] = {
    { XMFLOAT3(-1.0f,  1.0f, 0.0f), XMFLOAT2(0.0f, 0.0f) },
    { XMFLOAT3(1.0f,  1.0f, 0.0f), XMFLOAT2(1.0f, 0.0f) },
    { XMFLOAT3(1.0f, -1.0f, 0.0f), XMFLOAT2(1.0f, 1.0f) },
    { XMFLOAT3(-1.0f, -1.0f, 0.0f), XMFLOAT2(0.0f, 1.0f) },
};

DWORD indices[] = {
    0, 1, 2,
    0, 2, 3,
};

// Forward declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
HRESULT InitD3D(HWND hWnd);
void CleanD3D();
void RenderFrame();
void InitGraphics();
void InitPipeline();
void UpdateConstantBuffer();
void CompileShaderFromMemory(const char* shaderCode, const char* entryPoint, const char* profile, ID3DBlob** blob);
HRESULT LoadTextureFromFile(const wchar_t* fileName, ID3D11ShaderResourceView** textureView);

HRESULT LoadTextureFromFile(const wchar_t* fileName, ID3D11ShaderResourceView** textureView) {
    // Use the dds.c library to load the DDS texture
    // Make sure to include the necessary headers and link against the library

    FILE* file = _wfopen(fileName, L"rb");
    if (!file) {
        return E_FAIL;
    }

    DDS_HEADER header;
    fread(&header, sizeof(DDS_HEADER), 1, file);

    // Read the texture data and create a D3D11 texture
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = header.dwWidth;
    textureDesc.Height = header.dwHeight;
    textureDesc.MipLevels = header.dwMipMapCount;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // or appropriate format based on header
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_IMMUTABLE;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    // Read pixel data
    size_t dataSize = header.dwSize; // Read appropriate data size based on header
    BYTE* data = new BYTE[dataSize];
    fread(data, 1, dataSize, file);
    fclose(file);

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = data;
    initData.SysMemPitch = header.dwWidth * 4; // Assuming 4 bytes per pixel
    initData.SysMemSlicePitch = 0;

    HRESULT hr = gDevice->CreateTexture2D(&textureDesc, &initData, &gTexture);
    delete[] data;

    if (SUCCEEDED(hr)) {
        hr = gDevice->CreateShaderResourceView(gTexture, NULL, textureView);
    }
    return hr;
}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"D3D11VideoWindowClass";
    RegisterClass(&wc);

    gHwnd = CreateWindowEx(NULL, L"D3D11VideoWindowClass", L"DirectX 11 Video Renderer", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, NULL, NULL, hInstance, NULL);
    ShowWindow(gHwnd, nCmdShow);

    if (FAILED(InitD3D(gHwnd))) return 0;

    MSG msg = { 0 };
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else {
            RenderFrame();
        }
    }

    CleanD3D();
    return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Initialize Direct3D
HRESULT InitD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC scd = { 0 };
    scd.BufferCount = 1;
    scd.BufferDesc.Width = 800;
    scd.BufferDesc.Height = 600;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hWnd;
    scd.SampleDesc.Count = 1;
    scd.Windowed = TRUE;

    if (FAILED(D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &scd, &gSwapChain, &gDevice, NULL, &gDeviceContext))) {
        return E_FAIL;
    }

    ID3D11Texture2D* backBuffer = nullptr;
    gSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&backBuffer);
    gDevice->CreateRenderTargetView(backBuffer, NULL, &gRenderTargetView);
    backBuffer->Release();

    gDeviceContext->OMSetRenderTargets(1, &gRenderTargetView, NULL);

    D3D11_VIEWPORT vp = { 0 };
    vp.Width = 800.0f;
    vp.Height = 600.0f;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    gDeviceContext->RSSetViewports(1, &vp);

    InitGraphics();
    InitPipeline();

    return S_OK;
}

void InitGraphics() {

    OutputDebugStringA("test");

    D3D11_BUFFER_DESC vbd = { 0 };
    vbd.Usage = D3D11_USAGE_DEFAULT;
    vbd.ByteWidth = sizeof(vertices);
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA vertexData = { 0 };
    vertexData.pSysMem = vertices;

    gDevice->CreateBuffer(&vbd, &vertexData, &gVertexBuffer);

    D3D11_BUFFER_DESC cbd = { 0 };
    cbd.Usage = D3D11_USAGE_DEFAULT;
    cbd.ByteWidth = sizeof(XMMATRIX);
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

    gDevice->CreateBuffer(&cbd, NULL, &gConstantBuffer);

    // Load a texture to simulate video frames
    LoadTextureFromFile(L"C:\\Windows\\Temp\\v2\\video_frame_001.dds", &gTextureView);

    OutputDebugStringA("test");
}

// Vertex and pixel shader code embedded in C array
const char* videoShaderCode = R"(
cbuffer ConstantBuffer : register(b0)
{
    matrix worldViewProj;
};

struct VS_INPUT
{
    float4 pos : POSITION;
    float2 tex : TEXCOORD;
};

struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD;
};

PS_INPUT VSMain(VS_INPUT input)
{
    PS_INPUT output;
    output.pos = mul(input.pos, worldViewProj);
    output.tex = input.tex;
    return output;
}

Texture2D myTexture : register(t0);
SamplerState mySampler : register(s0);

float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return myTexture.Sample(mySampler, input.tex);
}
)";

void InitPipeline() {

    if (gDevice == NULL) 
        return;

    ID3DBlob* vsBlob = nullptr;
    CompileShaderFromMemory(videoShaderCode, "VSMain", "vs_4_0", &vsBlob);
    gDevice->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), NULL, &gVertexShader);

    ID3DBlob* psBlob = nullptr;
    CompileShaderFromMemory(videoShaderCode, "PSMain", "ps_4_0", &psBlob);
    gDevice->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), NULL, &gPixelShader);

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    gDevice->CreateInputLayout(layout, ARRAYSIZE(layout), vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &gInputLayout);
    gDeviceContext->IASetInputLayout(gInputLayout);

    vsBlob->Release();
    psBlob->Release();

    // Create sampler state for texture sampling
    D3D11_SAMPLER_DESC sd = {};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    gDevice->CreateSamplerState(&sd, &gSamplerState);

    gDeviceContext->PSSetSamplers(0, 1, &gSamplerState);
}

void RenderFrame() {
    float bgColor[4] = { 0.0f, 0.2f, 0.4f, 1.0f };
    gDeviceContext->ClearRenderTargetView(gRenderTargetView, bgColor);

    UpdateConstantBuffer();

    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    gDeviceContext->IASetVertexBuffers(0, 1, &gVertexBuffer, &stride, &offset);
    gDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    gDeviceContext->VSSetShader(gVertexShader, NULL, 0);
    gDeviceContext->PSSetShader(gPixelShader, NULL, 0);

    gDeviceContext->PSSetShaderResources(0, 1, &gTextureView);

    gDeviceContext->DrawIndexed(6, 0, 0);

    gSwapChain->Present(1, 0);
}

void UpdateConstantBuffer() {
    XMMATRIX world = XMMatrixIdentity();
    XMMATRIX view = XMMatrixLookAtLH(XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f), XMVectorZero(), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));
    XMMATRIX proj = XMMatrixOrthographicLH(2.0f, 2.0f, 0.1f, 100.0f);
    XMMATRIX wvp = world * view * proj;

    gDeviceContext->UpdateSubresource(gConstantBuffer, 0, NULL, &wvp, 0, 0);
    gDeviceContext->VSSetConstantBuffers(0, 1, &gConstantBuffer);
}

void CleanD3D() {
    if (gSamplerState) gSamplerState->Release();
    if (gTextureView) gTextureView->Release();
    if (gVertexBuffer) gVertexBuffer->Release();
    if (gConstantBuffer) gConstantBuffer->Release();
    if (gVertexShader) gVertexShader->Release();
    if (gPixelShader) gPixelShader->Release();
    if (gInputLayout) gInputLayout->Release();
    if (gRenderTargetView) gRenderTargetView->Release();
    if (gSwapChain) gSwapChain->Release();
    if (gDevice) gDevice->Release();
    if (gDeviceContext) gDeviceContext->Release();
}

//HRESULT LoadTextureFromFile(const wchar_t* fileName, ID3D11ShaderResourceView** textureView) {
//    // You can use DirectXTex or other libraries to load the video texture here.
//    return S_OK;
//}

void CompileShaderFromMemory(const char* shaderCode, const char* entryPoint, const char* profile, ID3DBlob** blob) {
    ID3DBlob* errorBlob = nullptr;
    D3DCompile(shaderCode, strlen(shaderCode), NULL, NULL, NULL, entryPoint, profile, 0, 0, blob, &errorBlob);
    if (errorBlob) {
        OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        errorBlob->Release();
    }
}
#endif





// rotation cube shader 4_0
#if 0 
#include <windows.h>
#include <d3d11.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;

// Define vertex structure
struct Vertex {
    XMFLOAT3 position;
    XMFLOAT3 color;
};

// Constant buffer for passing the transformation matrix
struct ConstantBuffer {
    XMMATRIX worldViewProj;
};

// Global variables for Direct3D
IDXGISwapChain* gSwapChain = nullptr;
ID3D11Device* gDevice = nullptr;
ID3D11DeviceContext* gDeviceContext = nullptr;
ID3D11RenderTargetView* gRenderTargetView = nullptr;
ID3D11Buffer* gVertexBuffer = nullptr;
ID3D11Buffer* gIndexBuffer = nullptr;
ID3D11Buffer* gConstantBuffer = nullptr;
ID3D11VertexShader* gVertexShader = nullptr;
ID3D11PixelShader* gPixelShader = nullptr;
ID3D11InputLayout* gInputLayout = nullptr;

float gRotationAngle = 0.0f;
HWND gHwnd = nullptr;

// Vertices of a cube (with colors)
Vertex vertices[] = {
    { XMFLOAT3(-1.0f, -1.0f, -1.0f), XMFLOAT3(1.0f, 0.0f, 0.0f) }, // Red
    { XMFLOAT3(-1.0f,  1.0f, -1.0f), XMFLOAT3(0.0f, 1.0f, 0.0f) }, // Green
    { XMFLOAT3(1.0f,  1.0f, -1.0f), XMFLOAT3(0.0f, 0.0f, 1.0f) }, // Blue
    { XMFLOAT3(1.0f, -1.0f, -1.0f), XMFLOAT3(1.0f, 1.0f, 0.0f) }, // Yellow
    { XMFLOAT3(-1.0f, -1.0f,  1.0f), XMFLOAT3(1.0f, 0.0f, 1.0f) }, // Magenta
    { XMFLOAT3(-1.0f,  1.0f,  1.0f), XMFLOAT3(0.0f, 1.0f, 1.0f) }, // Cyan
    { XMFLOAT3(1.0f,  1.0f,  1.0f), XMFLOAT3(1.0f, 1.0f, 1.0f) }, // White
    { XMFLOAT3(1.0f, -1.0f,  1.0f), XMFLOAT3(0.5f, 0.5f, 0.5f) }, // Gray
};

// Cube indices
DWORD indices[] = {
    0, 1, 2, 0, 2, 3, // Back face
    4, 6, 5, 4, 7, 6, // Front face
    4, 5, 1, 4, 1, 0, // Left face
    3, 2, 6, 3, 6, 7, // Right face
    1, 5, 6, 1, 6, 2, // Top face
    4, 0, 3, 4, 3, 7, // Bottom face
};

// Forward declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
HRESULT InitD3D(HWND hWnd);
void CleanD3D();
void RenderFrame();
void InitGraphics();
void InitPipeline();
void UpdateConstantBuffer();
void CompileShaderFromMemory(const char* shaderCode, const char* entryPoint, const char* profile, ID3DBlob** blob);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    // Register the window class
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"D3D11WindowClass";
    RegisterClass(&wc);

    // Create the window
    gHwnd = CreateWindowEx(NULL, L"D3D11WindowClass", L"DirectX 11 Rotating Cube", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, NULL, NULL, hInstance, NULL);
    ShowWindow(gHwnd, nCmdShow);

    // Initialize Direct3D
    if (FAILED(InitD3D(gHwnd))) return 0;

    // Main message loop
    MSG msg = { 0 };
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else {
            RenderFrame();
        }
    }

    // Clean up Direct3D
    CleanD3D();
    return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Vertex and pixel shader code embedded in C array
const char* shaderCode = R"(
cbuffer ConstantBuffer : register(b0)
{
    matrix worldViewProj;
};

struct VS_INPUT
{
    float4 pos : POSITION;
    float4 color : COLOR;
};

struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};

PS_INPUT VSMain(VS_INPUT input)
{
    PS_INPUT output;
    output.pos = mul(input.pos, worldViewProj);
    output.color = input.color;
    return output;
}

float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return input.color;
}
)";

// Initialize Direct3D
HRESULT InitD3D(HWND hWnd) {
    // Swap chain description
    DXGI_SWAP_CHAIN_DESC scd = { 0 };
    scd.BufferCount = 1;
    scd.BufferDesc.Width = 800;
    scd.BufferDesc.Height = 600;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hWnd;
    scd.SampleDesc.Count = 1;
    scd.Windowed = TRUE;

    // Create device, device context, and swap chain
    if (FAILED(D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &scd, &gSwapChain, &gDevice, NULL, &gDeviceContext))) {
        return E_FAIL;
    }

    // Get the back buffer and create the render target view
    ID3D11Texture2D* backBuffer = nullptr;
    gSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&backBuffer);
    gDevice->CreateRenderTargetView(backBuffer, NULL, &gRenderTargetView);
    backBuffer->Release();

    gDeviceContext->OMSetRenderTargets(1, &gRenderTargetView, NULL);

    // Set the viewport
    D3D11_VIEWPORT vp = { 0 };
    vp.Width = 800.0f;
    vp.Height = 600.0f;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    gDeviceContext->RSSetViewports(1, &vp);

    InitGraphics();
    InitPipeline();

    return S_OK;
}

// Initialize cube vertices and indices
void InitGraphics() {
    // Vertex buffer
    D3D11_BUFFER_DESC vbd = { 0 };
    vbd.Usage = D3D11_USAGE_DEFAULT;
    vbd.ByteWidth = sizeof(vertices);
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA vertexData = { 0 };
    vertexData.pSysMem = vertices;

    gDevice->CreateBuffer(&vbd, &vertexData, &gVertexBuffer);

    // Index buffer
    D3D11_BUFFER_DESC ibd = { 0 };
    ibd.Usage = D3D11_USAGE_DEFAULT;
    ibd.ByteWidth = sizeof(indices);
    ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;

    D3D11_SUBRESOURCE_DATA indexData = { 0 };
    indexData.pSysMem = indices;

    gDevice->CreateBuffer(&ibd, &indexData, &gIndexBuffer);
}

// Initialize shaders and input layout
void InitPipeline() {
    // Compile vertex shader (now using vs_4_0)
    ID3DBlob* vsBlob = nullptr;
    CompileShaderFromMemory(shaderCode, "VSMain", "vs_4_0", &vsBlob);
    gDevice->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), NULL, &gVertexShader);

    // Compile pixel shader (now using ps_4_0)
    ID3DBlob* psBlob = nullptr;
    CompileShaderFromMemory(shaderCode, "PSMain", "ps_4_0", &psBlob);
    gDevice->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), NULL, &gPixelShader);

    // Define the input layout
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    gDevice->CreateInputLayout(layout, ARRAYSIZE(layout), vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &gInputLayout);
    gDeviceContext->IASetInputLayout(gInputLayout);

    vsBlob->Release();
    psBlob->Release();

    // Create constant buffer
    D3D11_BUFFER_DESC cbd = { 0 };
    cbd.Usage = D3D11_USAGE_DEFAULT;
    cbd.ByteWidth = sizeof(ConstantBuffer);
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

    gDevice->CreateBuffer(&cbd, NULL, &gConstantBuffer);
    gDeviceContext->VSSetConstantBuffers(0, 1, &gConstantBuffer);
}

// Render a frame
void RenderFrame() {
    // Clear the back buffer
    float bgColor[4] = { 0.0f, 0.2f, 0.4f, 1.0f };
    gDeviceContext->ClearRenderTargetView(gRenderTargetView, bgColor);

    // Update constant buffer with rotation
    UpdateConstantBuffer();

    // Set buffers and draw cube
    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    gDeviceContext->IASetVertexBuffers(0, 1, &gVertexBuffer, &stride, &offset);
    gDeviceContext->IASetIndexBuffer(gIndexBuffer, DXGI_FORMAT_R32_UINT, 0);

    gDeviceContext->VSSetShader(gVertexShader, NULL, 0);
    gDeviceContext->PSSetShader(gPixelShader, NULL, 0);

    gDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    gDeviceContext->DrawIndexed(36, 0, 0);

    gSwapChain->Present(1, 0);
}

// Update the constant buffer with the rotation matrix
void UpdateConstantBuffer() {
    // Create rotation matrix and view/projection matrices
    XMMATRIX world = XMMatrixRotationY(gRotationAngle);
    XMMATRIX view = XMMatrixLookAtLH(XMVectorSet(0.0f, 2.0f, -5.0f, 0.0f), XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XM_PIDIV4, 800.0f / 600.0f, 0.1f, 100.0f);

    ConstantBuffer cb;
    cb.worldViewProj = XMMatrixTranspose(world * view * proj);

    gDeviceContext->UpdateSubresource(gConstantBuffer, 0, NULL, &cb, 0, 0);

    gRotationAngle += 0.01f;
}

// Clean up Direct3D
void CleanD3D() {
    if (gVertexBuffer) gVertexBuffer->Release();
    if (gIndexBuffer) gIndexBuffer->Release();
    if (gConstantBuffer) gConstantBuffer->Release();
    if (gVertexShader) gVertexShader->Release();
    if (gPixelShader) gPixelShader->Release();
    if (gInputLayout) gInputLayout->Release();
    if (gRenderTargetView) gRenderTargetView->Release();
    if (gSwapChain) gSwapChain->Release();
    if (gDevice) gDevice->Release();
    if (gDeviceContext) gDeviceContext->Release();
}

// Compile shader from memory
void CompileShaderFromMemory(const char* shaderCode, const char* entryPoint, const char* profile, ID3DBlob** blob) {
    ID3DBlob* errorBlob = nullptr;
    D3DCompile(shaderCode, strlen(shaderCode), NULL, NULL, NULL, entryPoint, profile, 0, 0, blob, &errorBlob);
    if (errorBlob) {
        OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        errorBlob->Release();
    }
}

#endif





#if 0
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <windowsx.h>

#include <d3d11.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

// DirectX 11 variables
ID3D11Device* g_pd3dDevice = NULL;
ID3D11DeviceContext* g_pImmediateContext = NULL;
IDXGISwapChain* g_pSwapChain = NULL;
ID3D11RenderTargetView* g_pRenderTargetView = NULL;
ID3D11Texture2D* g_pVideoTexture = NULL;
ID3D11ShaderResourceView* g_pTextureView = NULL;
ID3D11SamplerState* g_pSamplerLinear = NULL;

// Shaders and buffers
ID3D11VertexShader* g_pVertexShader = NULL;
ID3D11PixelShader* g_pPixelShader = NULL;
ID3D11InputLayout* g_pVertexLayout = NULL;
ID3D11Buffer* g_pVertexBuffer = NULL;
ID3D11Buffer* g_pIndexBuffer = NULL;

// Video frame data
unsigned char* g_pVideoFrameData = NULL;

// Window dimensions (update these accordingly)
int windowWidth = 800;
int windowHeight = 600;


HWND hStopButton;
HANDLE hThread;
volatile BOOL threadRunning = TRUE; // Control flag for the thread

typedef struct {
    char script[4096];       // Store the JavaScript code
    int pc;                  // Program counter for current line execution
    int inInfiniteLoop;      // Flag to detect infinite loop
    int running;             // Flag to check if the interpreter is running
    int debugLine;           // Current debug line
    char debugOutput[4096];  // Output buffer for the debug console
} JsInterpreter;

JsInterpreter js;

// Function to handle pseudo console.log
void console_log(const char* message) {
    strcat(js.debugOutput, "console.log: ");
    strcat(js.debugOutput, message);
    strcat(js.debugOutput, "\n");
}

// Very basic JS interpreter (emulating a pseudo-JS behavior)
void interpret_js(const char* js_code) {
    const char* p = js_code;
    js.pc = 0;
    js.inInfiniteLoop = 0;
    js.running = 1;

    // Max iterations to prevent an infinite loop in this simulation
#define MAX_ITERATIONS 10
    int counter = 0; // Counter for iterations
    int iterations = 0; // Count of how many iterations have been executed

    while (*p != '\0' && js.running) {
        // Check for console.log
        if (strncmp(p, "console.log", 11) == 0) {
            p += 11;
            while (*p != '(' && *p != '\0') p++;
            if (*p == '(') p++;
            const char* log_start = p;
            while (*p != ')' && *p != '\0') p++;
            char log_message[256];
            strncpy(log_message, log_start, p - log_start);
            log_message[p - log_start] = '\0';
            console_log(log_message);
            p++;  // Skip closing parenthesis
        }
        // Simulate a while(true) infinite loop detection

        else if (strncmp(p, "while(true)", 11) == 0) {
            js.inInfiniteLoop = 1;
            js.running = 0;  // Stop further execution
            console_log("Infinite loop detected, stopping execution...");
            break;
        }
        /*
        // Handle while(1) for counter simulation
        else if (strncmp(p, "while(1)", 9) == 0) {
            //p += 9; // Move past 'while(1)'
            js.inInfiniteLoop = 1;
            // Increment the counter and log the output for a limited number of iterations
            while (iterations < MAX_ITERATIONS) {
                counter++;
                char log_message[256];
                sprintf(log_message, "%d", counter); // Convert counter to string
                console_log(log_message); // Log the current counter value
                iterations++;
                Sleep(100); // Optional: Slow down the loop for demonstration purposes
            }
            console_log("Completed loop execution.");
            js.running = 0; // Stop further execution after the loop
            break;
        }
        */
        p++;
    }
}

DWORD WINAPI InfiniteLoopThread(LPVOID lpParam) {
    int counter = 0;
    while (threadRunning) {
        counter++;
        char log_message[256];
        sprintf(log_message, "Counter: %d\n", counter);
        console_log(log_message);
        Sleep(100); // Optional: Slow down the loop for demonstration
        InvalidateRect(NULL, NULL, TRUE); // repaint console
    }
    return 0;
}

void repaint_console() {
    // Assume you have a console control or text area in your application
    // For example, if you are using a static control for console output:
    //HWND hConsoleOutput = GetDlgItem(hwnd, IDC_CONSOLE_OUTPUT); // Replace IDC_CONSOLE_OUTPUT with your control ID

    // Clear the existing text in the console
    //SetWindowText(hConsoleOutput, "");

    //strcat(js.debugOutput, "console.log: "); js.debugOutput = "";
    //ZeroMemory(js.debugOutput, 4096);
    memset(js.debugOutput, 0, sizeof(js.debugOutput));

    //SetWindowText(hConsoleOutput, "");

    // Display updated messages
    // Example of showing all messages in a loop; customize as needed
    console_log("Current thread status: stopped"); // This will log the status
    // Additional messages you might want to show
    InvalidateRect(NULL, NULL, TRUE);  // Trigger repaint to show new debug output
}


// Function to start the JS interpreter and output debug messages
void start_js_interpreter() {
    memset(js.debugOutput, 0, sizeof(js.debugOutput));
    interpret_js(js.script);
    InvalidateRect(NULL, NULL, TRUE);  // Trigger repaint to show new debug output
}


typedef struct {
    COLORREF textColor;       // Text color
    COLORREF backgroundColor; // Background color
    int fontSize;             // Font size
} CssStyle;

typedef struct {
    char text[1024];
    int bold;
    int italic;
    CssStyle style;
    int isLink;             // 1 if this element is a link
    int isTable;            // 1 if this element belongs to a table
    RECT rect;              // Used for clickable areas (links, etc.)
    char href[256];         // Stores the link URL if it's a link
} HtmlElement;


// Convert a color string (like "#FF0000" for red) to COLORREF
COLORREF parse_color(const char* color) {
    if (color[0] == '#') {
        int r, g, b;
        sscanf(color, "#%02x%02x%02x", &r, &g, &b);
        return RGB(r, g, b);
    }
    // Default to black if no color specified
    return RGB(0, 0, 0);
}

// Very simple CSS parser for inline styles
void parse_css(const char* css, CssStyle* style) {
    const char* p = css;

    style->textColor = RGB(0, 0, 0);       // Default text color: black
    style->backgroundColor = RGB(255, 255, 255); // Default background: white
    style->fontSize = 20; // Default font size

    while (*p != '\0') {
        if (strncmp(p, "color:", 6) == 0) {
            p += 6;
            while (*p == ' ') p++;  // Skip spaces
            const char* color_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, color_start, p - color_start);
            color[p - color_start] = '\0';
            style->textColor = parse_color(color);
        }
        else if (strncmp(p, "background-color:", 17) == 0) {
            p += 17;
            while (*p == ' ') p++;  // Skip spaces
            const char* bg_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, bg_start, p - bg_start);
            color[p - bg_start] = '\0';
            style->backgroundColor = parse_color(color);
        }
        else if (strncmp(p, "font-size:", 10) == 0) {
            p += 10;
            while (*p == ' ') p++;  // Skip spaces
            int size = 0;
            sscanf(p, "%d", &size); // Parse font size
            style->fontSize = size;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
        }
        while (*p != ';' && *p != '\0') p++;  // Skip to next style attribute
        if (*p == ';') p++;
    }
}

void parse_html(const char* html, HtmlElement* elements, int* element_count) {
    const char* p = html;
    *element_count = 0;
    int isTable = 0;

    while (*p != '\0') {
        if (*p == '<') {
            if (strncmp(p, "<p>", 3) == 0) {
                p += 3;  // Skip <p> tag
                continue;
            }
            else if (strncmp(p, "</p>", 4) == 0) {
                p += 4;
                continue;
            }
            else if (strncmp(p, "<div", 4) == 0) {
                elements[*element_count].bold = 0;
                elements[*element_count].italic = 0;
                p += 4;
                // Handle inline styles like before
                continue;
            }
            else if (strncmp(p, "</div>", 6) == 0) {
                p += 6;
                continue;
            }
            // Parse <a href="..."> links
            else if (strncmp(p, "<a href=\"", 9) == 0) {
                p += 9; // Skip <a href="
                char* href_start = (char*)p;
                while (*p != '\"' && *p != '\0') p++; // Find the end of the href
                strncpy(elements[*element_count].href, href_start, p - href_start);
                elements[*element_count].href[p - href_start] = '\0';
                elements[*element_count].isLink = 1;
                p++;  // Skip closing "
                continue;
            }
            else if (strncmp(p, "</a>", 4) == 0) {
                p += 4;
                elements[*element_count].isLink = 0;  // Reset link flag after </a>
                continue;
            }
            // Parse table-related tags
            else if (strncmp(p, "<table>", 7) == 0) {
                p += 7;
                isTable = 1;  // Mark that we're in a table
                continue;
            }
            else if (strncmp(p, "</table>", 8) == 0) {
                p += 8;
                isTable = 0;  // We're leaving the table
                continue;
            }
            else if (strncmp(p, "<tr>", 4) == 0) {
                p += 4;  // New row
                continue;
            }
            else if (strncmp(p, "</tr>", 5) == 0) {
                p += 5;
                continue;
            }
            else if (strncmp(p, "<td>", 4) == 0) {
                p += 4;  // New cell
                elements[*element_count].isTable = 1;  // Mark this element as part of the table
                continue;
            }
            else if (strncmp(p, "</td>", 5) == 0) {
                p += 5;
                continue;
            }
        }
        else {
            // Read plain text
            char* text_start = (char*)p;
            while (*p != '<' && *p != '\0') {
                p++;
            }

            strncpy(elements[*element_count].text, text_start, p - text_start);
            elements[*element_count].text[p - text_start] = '\0';
            elements[*element_count].isTable = isTable;  // Assign table flag
            (*element_count)++;
        }
    }
}

// Vertex shader source code
const char* g_VS =
"cbuffer ConstantBuffer : register(b0) \
{\
    float4x4 mWorldViewProj;\
};\
struct VS_INPUT\
{\
    float3 Pos : POSITION;\
    float2 Tex : TEXCOORD0;\
};\
struct PS_INPUT\
{\
    float4 Pos : SV_POSITION;\
    float2 Tex : TEXCOORD0;\
};\
PS_INPUT VS(VS_INPUT input)\
{\
    PS_INPUT output = (PS_INPUT)0;\
    output.Pos = float4(input.Pos, 1.0);\
    output.Tex = input.Tex;\
    return output;\
}";

// Pixel shader source code
const char* g_PS =
"Texture2D txDiffuse : register(t0);\
SamplerState samLinear : register(s0);\
struct PS_INPUT\
{\
    float4 Pos : SV_POSITION;\
    float2 Tex : TEXCOORD0;\
};\
float4 PS(PS_INPUT input) : SV_Target\
{\
    return txDiffuse.Sample(samLinear, input.Tex);\
}";

HRESULT InitGraphics()
{
    HRESULT hr;

    // Compile and create vertex shader
    ID3DBlob* pVSBlob = NULL;
    hr = D3DCompile(g_VS, strlen(g_VS), NULL, NULL, NULL, "VS", "vs_4_0", 0, 0, &pVSBlob, NULL);
    if (FAILED(hr))
    {
        MessageBox(NULL, L"Failed to compile vertex shader.", L"Error", MB_OK);
        return hr;
    }

    hr = g_pd3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &g_pVertexShader);
    if (FAILED(hr))
    {
        pVSBlob->Release();
        return hr;
    }

    // Define input layout
    D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,                             D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, sizeof(float) * 3,             D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    hr = g_pd3dDevice->CreateInputLayout(layout, ARRAYSIZE(layout), pVSBlob->GetBufferPointer(),
        pVSBlob->GetBufferSize(), &g_pVertexLayout);
    pVSBlob->Release();
    if (FAILED(hr))
        return hr;

    g_pImmediateContext->IASetInputLayout(g_pVertexLayout);

    // Compile and create pixel shader
    ID3DBlob* pPSBlob = NULL;
    hr = D3DCompile(g_PS, strlen(g_PS), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, NULL);
    if (FAILED(hr))
    {
        MessageBox(NULL, L"Failed to compile pixel shader.", L"Error", MB_OK);
        return hr;
    }

    hr = g_pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &g_pPixelShader);
    pPSBlob->Release();
    if (FAILED(hr))
        return hr;

    // Create vertex buffer for a rectangle (quad)
    struct SimpleVertex
    {
        float Pos[3];
        float Tex[2];
    };

    // Coordinates for the rectangle (200x200 px at position (10,10))
    float left = -1.0f + 2.0f * 10 / windowWidth;
    float right = -1.0f + 2.0f * (10 + 200) / windowWidth;
    float top = 1.0f - 2.0f * 10 / windowHeight;
    float bottom = 1.0f - 2.0f * (10 + 200) / windowHeight;

    SimpleVertex vertices[] =
    {
        { { left,  top,    0.5f }, { 0.0f, 0.0f } },
        { { right, top,    0.5f }, { 1.0f, 0.0f } },
        { { right, bottom, 0.5f }, { 1.0f, 1.0f } },
        { { left,  bottom, 0.5f }, { 0.0f, 1.0f } },
    };

    D3D11_BUFFER_DESC bd = { };
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(SimpleVertex) * 4;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA InitData = { };
    InitData.pSysMem = vertices;

    hr = g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pVertexBuffer);
    if (FAILED(hr))
        return hr;

    // Create index buffer
    WORD indices[] = { 0,1,2, 2,3,0 };
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(WORD) * 6;
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    InitData.pSysMem = indices;

    hr = g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pIndexBuffer);
    if (FAILED(hr))
        return hr;

    // Create texture
    D3D11_TEXTURE2D_DESC texDesc = { };
    texDesc.Width = 640;
    texDesc.Height = 480;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DYNAMIC;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    hr = g_pd3dDevice->CreateTexture2D(&texDesc, NULL, &g_pVideoTexture);
    if (FAILED(hr))
        return hr;

    // Create shader resource view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
    srvDesc.Format = texDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = texDesc.MipLevels;
    hr = g_pd3dDevice->CreateShaderResourceView(g_pVideoTexture, &srvDesc, &g_pTextureView);
    if (FAILED(hr))
        return hr;

    // Create sampler state
    D3D11_SAMPLER_DESC sampDesc = { };
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = sampDesc.AddressV = sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    hr = g_pd3dDevice->CreateSamplerState(&sampDesc, &g_pSamplerLinear);
    if (FAILED(hr))
        return hr;

    // Allocate video frame data buffer
    //g_pVideoFrameData = (unsigned char*)malloc(640 * 480 * 4); // RGBA

    if (g_pVideoFrameData == nullptr)
    {
        // Allocate memory for the video frame (640x480 resolution with 4 bytes per pixel for RGBA)
        g_pVideoFrameData = new unsigned char[640 * 480 * 4];
    }

    return S_OK;
}

void UpdateTexture()
{
    // Generate random noise
    for (int i = 0; i < 640 * 480 * 4; i++)
    {
        g_pVideoFrameData[i] = rand() % 256;
    }

    // Update the texture
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    HRESULT hr = g_pImmediateContext->Map(g_pVideoTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    if (SUCCEEDED(hr))
    {
        memcpy(mappedResource.pData, g_pVideoFrameData, 640 * 480 * 4);
        g_pImmediateContext->Unmap(g_pVideoTexture, 0);
    }
}

void Render()
{
    // Update the texture
    UpdateTexture();

    // Clear the back buffer
    float ClearColor[4] = { 0.0f, 0.125f, 0.3f, 1.0f }; // RGBA
    g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView, ClearColor);

    // Set shaders
    g_pImmediateContext->VSSetShader(g_pVertexShader, NULL, 0);
    g_pImmediateContext->PSSetShader(g_pPixelShader, NULL, 0);

    // Set vertex buffer
    UINT stride = sizeof(float) * 5;
    UINT offset = 0;
    g_pImmediateContext->IASetVertexBuffers(0, 1, &g_pVertexBuffer, &stride, &offset);

    // Set index buffer
    g_pImmediateContext->IASetIndexBuffer(g_pIndexBuffer, DXGI_FORMAT_R16_UINT, 0);

    // Set primitive topology
    g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Set texture and sampler
    g_pImmediateContext->PSSetShaderResources(0, 1, &g_pTextureView);
    g_pImmediateContext->PSSetSamplers(0, 1, &g_pSamplerLinear);

    // Draw the quad
    g_pImmediateContext->DrawIndexed(6, 0, 0);

    // Present the buffer
    g_pSwapChain->Present(0, 0);
}


HRESULT InitD3D(HWND hWnd)
{

    OutputDebugStringW(L"test1");

    // Describe the swap chain
    DXGI_SWAP_CHAIN_DESC sd = { };
    sd.BufferCount = 1;
    sd.BufferDesc.Width = windowWidth;
    sd.BufferDesc.Height = windowHeight;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;    // No multi-sampling
    sd.Windowed = TRUE;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        NULL,                    // Default adapter
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        0,                       // Flags
        NULL,                    // Feature levels
        0,
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &g_pd3dDevice,
        NULL,
        &g_pImmediateContext
    );

    OutputDebugStringW(L"test2");

    if (FAILED(hr))
        return hr;

    // Create render target view
    ID3D11Texture2D* pBackBuffer = NULL;
    hr = g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    if (FAILED(hr))
        return hr;

    hr = g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_pRenderTargetView);
    pBackBuffer->Release();
    if (FAILED(hr))
        return hr;

    // Set the render target
    g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView, NULL);

    // Set the viewport
    D3D11_VIEWPORT vp;
    vp.Width = (FLOAT)windowWidth;
    vp.Height = (FLOAT)windowHeight;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pImmediateContext->RSSetViewports(1, &vp);

    OutputDebugStringW(L"testX");

    return S_OK;
}

void CleanupDevice()
{
    if (g_pImmediateContext) g_pImmediateContext->ClearState();
    if (g_pSamplerLinear) g_pSamplerLinear->Release();
    if (g_pTextureView) g_pTextureView->Release();
    if (g_pVideoTexture) g_pVideoTexture->Release();
    if (g_pVertexBuffer) g_pVertexBuffer->Release();
    if (g_pIndexBuffer) g_pIndexBuffer->Release();
    if (g_pVertexLayout) g_pVertexLayout->Release();
    if (g_pVertexShader) g_pVertexShader->Release();
    if (g_pPixelShader) g_pPixelShader->Release();
    if (g_pRenderTargetView) g_pRenderTargetView->Release();
    if (g_pSwapChain) g_pSwapChain->Release();
    if (g_pImmediateContext) g_pImmediateContext->Release();
    if (g_pd3dDevice) g_pd3dDevice->Release();
    if (g_pVideoFrameData) free(g_pVideoFrameData);
}

void RenderVideo()
{
    // Set viewport to the 200x200 area where the video will be rendered
    D3D11_VIEWPORT vp;
    vp.Width = 600.0f;
    vp.Height = 400.0f;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 100;  // Adjust as per your layout
    vp.TopLeftY = 100;
    g_pImmediateContext->RSSetViewports(1, &vp);

    // Update and render the video texture (as shown in previous steps)
    UpdateTexture();  // Update the random noise texture
    Render();         // Render the texture using DirectX
}


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    static HtmlElement elements[100];
    static int element_count = 0;

    const char* html_content =
        "<div style=\"color:#FF0000;\">This is a div</div>"
        "<table>"
        "<tr><td>Row 1, Col 1</td><td>Row 1, Col 2</td></tr>"
        "<tr><td>Row 2, Col 1</td><td>Row 2, Col 2</td></tr>"
        "</table>"
        "<a href=\"http://example.com\">Click here for example</a>";

    strcpy(js.script, "console.log('Starting JS execution'); while(true);");
    //strcpy(js.script, "while(1) { counter=0; console.log(counter); counter+=1; }");



    switch (uMsg)
    {
    case WM_CREATE:
    {

        parse_html(html_content, elements, &element_count);

        //InitGraphics();

        //InitD3D(hwnd);

         // Set up a 16ms timer for refreshing the video (60 fps)
        SetTimer(hwnd, 1, 1000, NULL);

    }
    break;

    case WM_COMMAND:
    {
        if (LOWORD(wParam) == 1) {  // "Run JS" button was clicked
            start_js_interpreter();

            threadRunning = TRUE;
            // Start the infinite loop thread
            hThread = CreateThread(NULL, 0, InfiniteLoopThread, NULL, 0, NULL);
        }

        // Check if the Stop Thread button was clicked
        if (LOWORD(wParam) == BN_CLICKED && (HWND)lParam == hStopButton) {
            threadRunning = FALSE;  // Signal the thread to stop
            WaitForSingleObject(hThread, INFINITE); // Wait for the thread to finish
            CloseHandle(hThread); // Clean up the thread handle
            hThread = NULL; // Reset thread handle

            repaint_console(); // Call to repaint console on F5

            console_log("Thread stopped.");
        }

    }
    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        LOGFONTW logfont = { 0 };

        int x = 10;
        int y = 10;

        int table_row_start = x;
        int cell_width = 200;
        int cell_height = 50;
        for (int i = 0; i < element_count; i++) {
            if (elements[i].isTable) {
                // Draw table cell borders
                RECT cell_rect = { x, y, x + cell_width, y + cell_height };
                DrawEdge(hdc, &cell_rect, EDGE_RAISED, BF_RECT);

                // Set font and styles like before
                HFONT hFont = CreateFontIndirect(&logfont);
                SelectObject(hdc, hFont);
                SetTextColor(hdc, elements[i].style.textColor);

                // Draw text inside the cell
                WCHAR wText[1024];
                MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024);
                TextOutW(hdc, x + 10, y + 10, wText, wcslen(wText));

                DeleteObject(hFont);

                // Move to the next cell
                x += cell_width;
            }
            else {
                // For non-table elements, draw them normally
                HFONT hFont = CreateFontIndirect(&logfont);
                SelectObject(hdc, hFont);
                SetTextColor(hdc, elements[i].style.textColor);

                // Check if this is a link
                if (elements[i].isLink) {
                    elements[i].rect = { x, y, x + 200, y + 20 };  // Define clickable area
                    SetTextColor(hdc, RGB(0, 0, 255));  // Links are typically blue
                    // Optionally underline the link
                    logfont.lfUnderline = TRUE;
                }

                WCHAR wText[1024];
                MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024);
                TextOutW(hdc, x, y, wText, wcslen(wText));

                DeleteObject(hFont);
                y += 30;
            }
        }

        const int pos_X_console = 600;
        const int pos_X1_console = 1000;
        const int pos_Y_console = 100;

        // Right side: Render Debug console
        RECT debugRect = { pos_X_console, pos_Y_console, pos_X1_console, 600 };  // Define right-side rectangle for debug console
        HBRUSH hBrush = CreateSolidBrush(RGB(240, 240, 240)); // Light gray background for console
        FillRect(hdc, &debugRect, hBrush);
        DeleteObject(hBrush);

        // Draw border for debug console
        DrawEdge(hdc, &debugRect, EDGE_RAISED, BF_RECT);

        // Set font for debug console output
        HFONT hFont = CreateFont(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
            DEFAULT_PITCH | FF_SWISS, L"Consolas");
        SelectObject(hdc, hFont);

        SetTextColor(hdc, RGB(0, 0, 0));  // Black text
        SetBkMode(hdc, TRANSPARENT);      // Transparent background for text

        // Draw debug output (like a console)
        RECT textRect = { pos_X_console + 10, pos_Y_console + 10, pos_X1_console - 10, 590 };

        WCHAR wText[1024];
        MultiByteToWideChar(CP_ACP, 0, js.debugOutput, -1, wText, 1024);

        OutputDebugStringW((LPCWSTR)js.debugOutput);

        DrawText(hdc, wText, -1, &textRect, DT_LEFT | DT_TOP | DT_WORDBREAK);

        // Now render video inside the 200x200 rectangle
        RenderVideo();

        DeleteObject(hFont);
        EndPaint(hwnd, &ps);
    }
    break;

    case WM_TIMER:
    {
        // Invalidate only the 200x200 region where the video is displayed
        RECT videoRect = { 10, 10, 410, 410 }; // Adjust according to your video position
        InvalidateRect(hwnd, &videoRect, FALSE);
        //InvalidateRect(NULL, NULL, TRUE);
    }
    break;

    case WM_DESTROY:

        CleanupDevice();

        // Cleanup thread if still running
        if (threadRunning) {
            threadRunning = FALSE; // Stop the thread if running
            WaitForSingleObject(hThread, INFINITE); // Wait for the thread to finish
            CloseHandle(hThread); // Clean up the thread handle
        }

        KillTimer(hwnd, 1); // Stop the timer
        PostQuitMessage(0);
        return 0;

    case WM_SIZE:
    {
        windowWidth = LOWORD(lParam);
        windowHeight = HIWORD(lParam);
        // Handle resizing DirectX resources if needed
    }
    break;

    case WM_LBUTTONDOWN:
    {
        int xPos = GET_X_LPARAM(lParam);
        int yPos = GET_Y_LPARAM(lParam);

        for (int i = 0; i < element_count; i++) {
            if (elements[i].isLink && PtInRect(&elements[i].rect, { xPos, yPos })) {
                // Handle link click - for example, simulate navigation
                MessageBoxA(hwnd, elements[i].href, "Clicked Link", MB_OK);
            }
        }

        // Check if the click is on a specific button area (e.g., "Run JS")
        if (xPos >= 620 && xPos <= 720 && yPos >= 620 && yPos <= 650) {
            // Start the JS interpreter when "Run JS" button is clicked
            start_js_interpreter();
        }

    }
    break;

    case WM_SETCURSOR:
    {
        POINT pt;
        GetCursorPos(&pt);
        ScreenToClient(hwnd, &pt);

        for (int i = 0; i < element_count; i++) {
            if (elements[i].isLink && PtInRect(&elements[i].rect, pt)) {
                SetCursor(LoadCursor(NULL, IDC_HAND));
                return TRUE;  // We handled the cursor change
            }
        }
        SetCursor(LoadCursor(NULL, IDC_ARROW));
        return TRUE;
    }

    case WM_KEYDOWN:
    {
        // Check if the F5 key was pressed
        if (wParam == VK_F5) {
            OutputDebugStringW(L"refresh");
            console_log("Refreshing application...");

            // Reparse the HTML content
            parse_html(html_content, elements, &element_count);
            InvalidateRect(NULL, NULL, TRUE);  // Trigger to repaint console
        }
    }
    break;


    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Register the window class
    const char CLASS_NAME[] = "Sample Window Class";

    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCWSTR)CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles.
        (LPCWSTR)CLASS_NAME,                     // Window class
        L"Simple Browser",               // Window text
        WS_OVERLAPPEDWINDOW,            // Window style

        // Position and size
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL)
    {
        return 0;
    }

    // Create a button to trigger "Run JS"
    CreateWindow(L"BUTTON", L"Run JS", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON, 10, 400, 100, 30, hwnd, (HMENU)1, hInstance, NULL);

    // Create Stop Thread button
    hStopButton = CreateWindow(
        L"BUTTON",  // Predefined class; Unicode assumed 
        L"Stop Thread",      // Button text 
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles 
        140,         // x position 
        400,         // y position 
        100,        // Button width
        30,         // Button height
        hwnd,     // Parent window
        NULL,       // No menu.
        (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE),
        NULL);      // Pointer not needed.

    ShowWindow(hwnd, nCmdShow);

    // init dx
    
    InitD3D(hwnd);
    
    InitGraphics();

    // start timer - Start the timer (maybe set a 16ms interval for ~60fps):
    //SetTimer(hwnd, 1, 16, NULL); // 16ms timer for 60 FPS


    // Run the message loop
    MSG msg = { };
    
    /*
    while (GetMessage(&msg, NULL, 0, 0))
    {
         TranslateMessage(&msg);
         DispatchMessage(&msg);
         Render();
       
    }
    */

    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            // render();
            // We don't render in the loop directly anymore
            // Allow WM_PAINT to trigger the rendering.
        }
    }

    return 0;
}
#endif
















// this not working - console not working
#if 0
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <windowsx.h>

HWND hStopButton;
HANDLE hThread;
volatile BOOL threadRunning = TRUE; // Control flag for the thread

typedef struct {
    char script[4096];       // Store the JavaScript code
    int pc;                  // Program counter for current line execution
    int inInfiniteLoop;      // Flag to detect infinite loop
    int running;             // Flag to check if the interpreter is running
    int debugLine;           // Current debug line
    char debugOutput[4096];  // Output buffer for the debug console
} JsInterpreter;

JsInterpreter js;

// Function to handle pseudo console.log
void console_log(const char* message) {
    strcat(js.debugOutput, "console.log: ");
    strcat(js.debugOutput, message);
    strcat(js.debugOutput, "\n");
}

// Very basic JS interpreter (emulating a pseudo-JS behavior)
void interpret_js(const char* js_code) {
    const char* p = js_code;
    js.pc = 0;
    js.inInfiniteLoop = 0;
    js.running = 1;

    // Max iterations to prevent an infinite loop in this simulation
#define MAX_ITERATIONS 10
    int counter = 0; // Counter for iterations
    int iterations = 0; // Count of how many iterations have been executed

    while (*p != '\0' && js.running) {
        // Check for console.log
        if (strncmp(p, "console.log", 11) == 0) {
            p += 11;
            while (*p != '(' && *p != '\0') p++;
            if (*p == '(') p++;
            const char* log_start = p;
            while (*p != ')' && *p != '\0') p++;
            char log_message[256];
            strncpy(log_message, log_start, p - log_start);
            log_message[p - log_start] = '\0';
            console_log(log_message);
            p++;  // Skip closing parenthesis
        }
        // Simulate a while(true) infinite loop detection
        
        else if (strncmp(p, "while(true)", 11) == 0) {
            js.inInfiniteLoop = 1;
            js.running = 0;  // Stop further execution
            console_log("Infinite loop detected, stopping execution...");
            break;
        }
        /*
        // Handle while(1) for counter simulation
        else if (strncmp(p, "while(1)", 9) == 0) {
            //p += 9; // Move past 'while(1)'
            js.inInfiniteLoop = 1;
            // Increment the counter and log the output for a limited number of iterations
            while (iterations < MAX_ITERATIONS) {
                counter++;
                char log_message[256];
                sprintf(log_message, "%d", counter); // Convert counter to string
                console_log(log_message); // Log the current counter value
                iterations++;
                Sleep(100); // Optional: Slow down the loop for demonstration purposes
            }
            console_log("Completed loop execution.");
            js.running = 0; // Stop further execution after the loop
            break;
        }
        */
        p++;
    }
}

DWORD WINAPI InfiniteLoopThread(LPVOID lpParam) {
    int counter = 0;
    while (threadRunning) {
        counter++;
        char log_message[256];
        sprintf(log_message, "Counter: %d\n", counter);
        console_log(log_message);
        Sleep(100); // Optional: Slow down the loop for demonstration
        InvalidateRect(NULL, NULL, TRUE); // repaint console
    }
    return 0;
}

void repaint_console() {
    // Assume you have a console control or text area in your application
    // For example, if you are using a static control for console output:
    //HWND hConsoleOutput = GetDlgItem(hwnd, IDC_CONSOLE_OUTPUT); // Replace IDC_CONSOLE_OUTPUT with your control ID

    // Clear the existing text in the console
    //SetWindowText(hConsoleOutput, "");

    //strcat(js.debugOutput, "console.log: "); js.debugOutput = "";
    //ZeroMemory(js.debugOutput, 4096);
    memset(js.debugOutput, 0, sizeof(js.debugOutput));

    //SetWindowText(hConsoleOutput, "");

    // Display updated messages
    // Example of showing all messages in a loop; customize as needed
    console_log("Current thread status: stopped"); // This will log the status
    // Additional messages you might want to show
    InvalidateRect(NULL, NULL, TRUE);  // Trigger repaint to show new debug output
}


// Function to start the JS interpreter and output debug messages
void start_js_interpreter() {
    memset(js.debugOutput, 0, sizeof(js.debugOutput));
    interpret_js(js.script);
    InvalidateRect(NULL, NULL, TRUE);  // Trigger repaint to show new debug output
}


typedef struct {
    COLORREF textColor;       // Text color
    COLORREF backgroundColor; // Background color
    int fontSize;             // Font size
} CssStyle;

typedef struct {
    char text[1024];
    int bold;
    int italic;
    CssStyle style;
    int isLink;             // 1 if this element is a link
    int isTable;            // 1 if this element belongs to a table
    RECT rect;              // Used for clickable areas (links, etc.)
    char href[256];         // Stores the link URL if it's a link
} HtmlElement;


// Convert a color string (like "#FF0000" for red) to COLORREF
COLORREF parse_color(const char* color) {
    if (color[0] == '#') {
        int r, g, b;
        sscanf(color, "#%02x%02x%02x", &r, &g, &b);
        return RGB(r, g, b);
    }
    // Default to black if no color specified
    return RGB(0, 0, 0);
}

// Very simple CSS parser for inline styles
void parse_css(const char* css, CssStyle* style) {
    const char* p = css;

    style->textColor = RGB(0, 0, 0);       // Default text color: black
    style->backgroundColor = RGB(255, 255, 255); // Default background: white
    style->fontSize = 20; // Default font size

    while (*p != '\0') {
        if (strncmp(p, "color:", 6) == 0) {
            p += 6;
            while (*p == ' ') p++;  // Skip spaces
            const char* color_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, color_start, p - color_start);
            color[p - color_start] = '\0';
            style->textColor = parse_color(color);
        }
        else if (strncmp(p, "background-color:", 17) == 0) {
            p += 17;
            while (*p == ' ') p++;  // Skip spaces
            const char* bg_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, bg_start, p - bg_start);
            color[p - bg_start] = '\0';
            style->backgroundColor = parse_color(color);
        }
        else if (strncmp(p, "font-size:", 10) == 0) {
            p += 10;
            while (*p == ' ') p++;  // Skip spaces
            int size = 0;
            sscanf(p, "%d", &size); // Parse font size
            style->fontSize = size;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
        }
        while (*p != ';' && *p != '\0') p++;  // Skip to next style attribute
        if (*p == ';') p++;
    }
}

void parse_html(const char* html, HtmlElement* elements, int* element_count) {
    const char* p = html;
    *element_count = 0;
    int isTable = 0;

    while (*p != '\0') {
        if (*p == '<') {
            if (strncmp(p, "<p>", 3) == 0) {
                p += 3;  // Skip <p> tag
                continue;
            }
            else if (strncmp(p, "</p>", 4) == 0) {
                p += 4;
                continue;
            }
            else if (strncmp(p, "<div", 4) == 0) {
                elements[*element_count].bold = 0;
                elements[*element_count].italic = 0;
                p += 4;
                // Handle inline styles like before
                continue;
            }
            else if (strncmp(p, "</div>", 6) == 0) {
                p += 6;
                continue;
            }
            // Parse <a href="..."> links
            else if (strncmp(p, "<a href=\"", 9) == 0) {
                p += 9; // Skip <a href="
                char* href_start = (char*)p;
                while (*p != '\"' && *p != '\0') p++; // Find the end of the href
                strncpy(elements[*element_count].href, href_start, p - href_start);
                elements[*element_count].href[p - href_start] = '\0';
                elements[*element_count].isLink = 1;
                p++;  // Skip closing "
                continue;
            }
            else if (strncmp(p, "</a>", 4) == 0) {
                p += 4;
                elements[*element_count].isLink = 0;  // Reset link flag after </a>
                continue;
            }
            // Parse table-related tags
            else if (strncmp(p, "<table>", 7) == 0) {
                p += 7;
                isTable = 1;  // Mark that we're in a table
                continue;
            }
            else if (strncmp(p, "</table>", 8) == 0) {
                p += 8;
                isTable = 0;  // We're leaving the table
                continue;
            }
            else if (strncmp(p, "<tr>", 4) == 0) {
                p += 4;  // New row
                continue;
            }
            else if (strncmp(p, "</tr>", 5) == 0) {
                p += 5;
                continue;
            }
            else if (strncmp(p, "<td>", 4) == 0) {
                p += 4;  // New cell
                elements[*element_count].isTable = 1;  // Mark this element as part of the table
                continue;
            }
            else if (strncmp(p, "</td>", 5) == 0) {
                p += 5;
                continue;
            }
        }
        else {
            // Read plain text
            char* text_start = (char*)p;
            while (*p != '<' && *p != '\0') {
                p++;
            }

            strncpy(elements[*element_count].text, text_start, p - text_start);
            elements[*element_count].text[p - text_start] = '\0';
            elements[*element_count].isTable = isTable;  // Assign table flag
            (*element_count)++;
        }
    }
}


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    static HtmlElement elements[100];
    static int element_count = 0;

    const char* html_content =
        "<div style=\"color:#FF0000;\">This is a div</div>"
        "<table>"
        "<tr><td>Row 1, Col 1</td><td>Row 1, Col 2</td></tr>"
        "<tr><td>Row 2, Col 1</td><td>Row 2, Col 2</td></tr>"
        "</table>"
        "<a href=\"http://example.com\">Click here for example</a>";

    strcpy(js.script, "console.log('Starting JS execution'); while(true);");
    //strcpy(js.script, "while(1) { counter=0; console.log(counter); counter+=1; }");



    switch (uMsg)
    {
    case WM_CREATE:
    {
        parse_html(html_content, elements, &element_count);

    }
    break;

    case WM_COMMAND:
    {
        if (LOWORD(wParam) == 1) {  // "Run JS" button was clicked
            start_js_interpreter();

            threadRunning = TRUE;
            // Start the infinite loop thread
            hThread = CreateThread(NULL, 0, InfiniteLoopThread, NULL, 0, NULL);
        }
    
        // Check if the Stop Thread button was clicked
        if (LOWORD(wParam) == BN_CLICKED && (HWND)lParam == hStopButton) {
            threadRunning = FALSE;  // Signal the thread to stop
            WaitForSingleObject(hThread, INFINITE); // Wait for the thread to finish
            CloseHandle(hThread); // Clean up the thread handle
            hThread = NULL; // Reset thread handle

            repaint_console(); // Call to repaint console on F5

            console_log("Thread stopped.");
        }
    
    }
    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        LOGFONTW logfont = { 0 };

        int x = 10;
        int y = 10;

        int table_row_start = x;
        int cell_width = 200;
        int cell_height = 50;
        for (int i = 0; i < element_count; i++) {
            if (elements[i].isTable) {
                // Draw table cell borders
                RECT cell_rect = { x, y, x + cell_width, y + cell_height };
                DrawEdge(hdc, &cell_rect, EDGE_RAISED, BF_RECT);

                // Set font and styles like before
                HFONT hFont = CreateFontIndirect(&logfont);
                SelectObject(hdc, hFont);
                SetTextColor(hdc, elements[i].style.textColor);

                // Draw text inside the cell
                WCHAR wText[1024];
                MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024);
                TextOutW(hdc, x + 10, y + 10, wText, wcslen(wText));

                DeleteObject(hFont);

                // Move to the next cell
                x += cell_width;
            }
            else {
                // For non-table elements, draw them normally
                HFONT hFont = CreateFontIndirect(&logfont);
                SelectObject(hdc, hFont);
                SetTextColor(hdc, elements[i].style.textColor);

                // Check if this is a link
                if (elements[i].isLink) {
                    elements[i].rect = { x, y, x + 200, y + 20 };  // Define clickable area
                    SetTextColor(hdc, RGB(0, 0, 255));  // Links are typically blue
                    // Optionally underline the link
                    logfont.lfUnderline = TRUE;
                }

                WCHAR wText[1024];
                MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024);
                TextOutW(hdc, x, y, wText, wcslen(wText));

                DeleteObject(hFont);
                y += 30;
            }
        }

        const int pos_X_console = 600;
        const int pos_X1_console = 1000;
        const int pos_Y_console = 100;

        // Right side: Render Debug console
        RECT debugRect = { pos_X_console, pos_Y_console, pos_X1_console, 600 };  // Define right-side rectangle for debug console
        HBRUSH hBrush = CreateSolidBrush(RGB(240, 240, 240)); // Light gray background for console
        FillRect(hdc, &debugRect, hBrush);
        DeleteObject(hBrush);

        // Draw border for debug console
        DrawEdge(hdc, &debugRect, EDGE_RAISED, BF_RECT);

        // Set font for debug console output
        HFONT hFont = CreateFont(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
            DEFAULT_PITCH | FF_SWISS, L"Consolas");
        SelectObject(hdc, hFont);

        SetTextColor(hdc, RGB(0, 0, 0));  // Black text
        SetBkMode(hdc, TRANSPARENT);      // Transparent background for text

        // Draw debug output (like a console)
        RECT textRect = { pos_X_console + 10, pos_Y_console + 10, pos_X1_console - 10, 590 };

        WCHAR wText[1024];
        MultiByteToWideChar(CP_ACP, 0, js.debugOutput, -1, wText, 1024);

        OutputDebugStringW((LPCWSTR)js.debugOutput);
        
        DrawText(hdc, wText, -1, &textRect, DT_LEFT | DT_TOP | DT_WORDBREAK);

        DeleteObject(hFont);
        EndPaint(hwnd, &ps);
    }
    break;

    case WM_DESTROY:

        // Cleanup thread if still running
        if (threadRunning) {
            threadRunning = FALSE; // Stop the thread if running
            WaitForSingleObject(hThread, INFINITE); // Wait for the thread to finish
            CloseHandle(hThread); // Clean up the thread handle
        }

        PostQuitMessage(0);
        return 0;

    case WM_LBUTTONDOWN:
    {
        int xPos = GET_X_LPARAM(lParam);
        int yPos = GET_Y_LPARAM(lParam);

        for (int i = 0; i < element_count; i++) {
            if (elements[i].isLink && PtInRect(&elements[i].rect, { xPos, yPos })) {
                // Handle link click - for example, simulate navigation
                MessageBoxA(hwnd, elements[i].href, "Clicked Link", MB_OK);
            }
        }

        // Check if the click is on a specific button area (e.g., "Run JS")
        if (xPos >= 620 && xPos <= 720 && yPos >= 620 && yPos <= 650) {
            // Start the JS interpreter when "Run JS" button is clicked
            start_js_interpreter();
        }

    }
    break;

    case WM_SETCURSOR:
    {
        POINT pt;
        GetCursorPos(&pt);
        ScreenToClient(hwnd, &pt);

        for (int i = 0; i < element_count; i++) {
            if (elements[i].isLink && PtInRect(&elements[i].rect, pt)) {
                SetCursor(LoadCursor(NULL, IDC_HAND));
                return TRUE;  // We handled the cursor change
            }
        }
        SetCursor(LoadCursor(NULL, IDC_ARROW));
        return TRUE;
    }

    case WM_KEYDOWN:
    {
        // Check if the F5 key was pressed
        if (wParam == VK_F5) {
            OutputDebugStringW(L"refresh");
            console_log("Refreshing application...");

            // Reparse the HTML content
            parse_html(html_content, elements, &element_count);
            InvalidateRect(NULL, NULL, TRUE);  // Trigger to repaint console
        }
    }
    break;


    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Register the window class
    const char CLASS_NAME[] = "Sample Window Class";

    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCWSTR)CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles.
        (LPCWSTR)CLASS_NAME,                     // Window class
        L"Simple Browser",               // Window text
        WS_OVERLAPPEDWINDOW,            // Window style

        // Position and size
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL)
    {
        return 0;
    }

    // Create a button to trigger "Run JS"
    CreateWindow(L"BUTTON", L"Run JS", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON, 10, 400, 100, 30, hwnd, (HMENU)1, hInstance, NULL);

    // Create Stop Thread button
    hStopButton = CreateWindow(
       L"BUTTON",  // Predefined class; Unicode assumed 
       L"Stop Thread",      // Button text 
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles 
        140,         // x position 
        400,         // y position 
        100,        // Button width
        30,         // Button height
        hwnd,     // Parent window
        NULL,       // No menu.
        (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE),
        NULL);      // Pointer not needed.

    ShowWindow(hwnd, nCmdShow);

    // Run the message loop
    MSG msg = { };
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif


#if 0
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <windowsx.h>

typedef struct {
    COLORREF textColor;       // Text color
    COLORREF backgroundColor; // Background color
    int fontSize;             // Font size
} CssStyle;

typedef struct {
    char text[1024];
    int bold;
    int italic;
    CssStyle style;
    int isLink;             // 1 if this element is a link
    int isTable;            // 1 if this element belongs to a table
    RECT rect;              // Used for clickable areas (links, etc.)
    char href[256];         // Stores the link URL if it's a link
} HtmlElement;


// Convert a color string (like "#FF0000" for red) to COLORREF
COLORREF parse_color(const char* color) {
    if (color[0] == '#') {
        int r, g, b;
        sscanf(color, "#%02x%02x%02x", &r, &g, &b);
        return RGB(r, g, b);
    }
    // Default to black if no color specified
    return RGB(0, 0, 0);
}

// Very simple CSS parser for inline styles
void parse_css(const char* css, CssStyle* style) {
    const char* p = css;

    style->textColor = RGB(0, 0, 0);       // Default text color: black
    style->backgroundColor = RGB(255, 255, 255); // Default background: white
    style->fontSize = 20; // Default font size

    while (*p != '\0') {
        if (strncmp(p, "color:", 6) == 0) {
            p += 6;
            while (*p == ' ') p++;  // Skip spaces
            const char* color_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, color_start, p - color_start);
            color[p - color_start] = '\0';
            style->textColor = parse_color(color);
        }
        else if (strncmp(p, "background-color:", 17) == 0) {
            p += 17;
            while (*p == ' ') p++;  // Skip spaces
            const char* bg_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, bg_start, p - bg_start);
            color[p - bg_start] = '\0';
            style->backgroundColor = parse_color(color);
        }
        else if (strncmp(p, "font-size:", 10) == 0) {
            p += 10;
            while (*p == ' ') p++;  // Skip spaces
            int size = 0;
            sscanf(p, "%d", &size); // Parse font size
            style->fontSize = size;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
        }
        while (*p != ';' && *p != '\0') p++;  // Skip to next style attribute
        if (*p == ';') p++;
    }
}

void parse_html(const char* html, HtmlElement* elements, int* element_count) {
    const char* p = html;
    *element_count = 0;
    int isTable = 0;

    while (*p != '\0') {
        if (*p == '<') {
            if (strncmp(p, "<p>", 3) == 0) {
                p += 3;  // Skip <p> tag
                continue;
            }
            else if (strncmp(p, "</p>", 4) == 0) {
                p += 4;
                continue;
            }
            else if (strncmp(p, "<div", 4) == 0) {
                elements[*element_count].bold = 0;
                elements[*element_count].italic = 0;
                p += 4;
                // Handle inline styles like before
                continue;
            }
            else if (strncmp(p, "</div>", 6) == 0) {
                p += 6;
                continue;
            }
            // Parse <a href="..."> links
            else if (strncmp(p, "<a href=\"", 9) == 0) {
                p += 9; // Skip <a href="
                char* href_start = (char*)p;
                while (*p != '\"' && *p != '\0') p++; // Find the end of the href
                strncpy(elements[*element_count].href, href_start, p - href_start);
                elements[*element_count].href[p - href_start] = '\0';
                elements[*element_count].isLink = 1;
                p++;  // Skip closing "
                continue;
            }
            else if (strncmp(p, "</a>", 4) == 0) {
                p += 4;
                elements[*element_count].isLink = 0;  // Reset link flag after </a>
                continue;
            }
            // Parse table-related tags
            else if (strncmp(p, "<table>", 7) == 0) {
                p += 7;
                isTable = 1;  // Mark that we're in a table
                continue;
            }
            else if (strncmp(p, "</table>", 8) == 0) {
                p += 8;
                isTable = 0;  // We're leaving the table
                continue;
            }
            else if (strncmp(p, "<tr>", 4) == 0) {
                p += 4;  // New row
                continue;
            }
            else if (strncmp(p, "</tr>", 5) == 0) {
                p += 5;
                continue;
            }
            else if (strncmp(p, "<td>", 4) == 0) {
                p += 4;  // New cell
                elements[*element_count].isTable = 1;  // Mark this element as part of the table
                continue;
            }
            else if (strncmp(p, "</td>", 5) == 0) {
                p += 5;
                continue;
            }
        }
        else {
            // Read plain text
            char* text_start = (char*)p;
            while (*p != '<' && *p != '\0') {
                p++;
            }

            strncpy(elements[*element_count].text, text_start, p - text_start);
            elements[*element_count].text[p - text_start] = '\0';
            elements[*element_count].isTable = isTable;  // Assign table flag
            (*element_count)++;
        }
    }
}


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    static HtmlElement elements[100];
    static int element_count = 0;

    const char* html_content =
        "<div style=\"color:#FF0000;\">This is a div</div>"
        "<table>"
        "<tr><td>Row 1, Col 1</td><td>Row 1, Col 2</td></tr>"
        "<tr><td>Row 2, Col 1</td><td>Row 2, Col 2</td></tr>"
        "</table>"
        "<a href=\"http://example.com\">Click here for example</a>";

    switch (uMsg)
    {
    case WM_CREATE:
    {
        parse_html(html_content, elements, &element_count);
    }
    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        LOGFONTW logfont = { 0 };

        int x = 10;
        int y = 10;

        int table_row_start = x;
        int cell_width = 200;
        int cell_height = 50;
        for (int i = 0; i < element_count; i++) {
            if (elements[i].isTable) {
                // Draw table cell borders
                RECT cell_rect = { x, y, x + cell_width, y + cell_height };
                DrawEdge(hdc, &cell_rect, EDGE_RAISED, BF_RECT);

                // Set font and styles like before
                HFONT hFont = CreateFontIndirect(&logfont);
                SelectObject(hdc, hFont);
                SetTextColor(hdc, elements[i].style.textColor);

                // Draw text inside the cell
                WCHAR wText[1024];
                MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024);
                TextOutW(hdc, x + 10, y + 10, wText, wcslen(wText));

                DeleteObject(hFont);

                // Move to the next cell
                x += cell_width;
            }
            else {
                // For non-table elements, draw them normally
                HFONT hFont = CreateFontIndirect(&logfont);
                SelectObject(hdc, hFont);
                SetTextColor(hdc, elements[i].style.textColor);

                // Check if this is a link
                if (elements[i].isLink) {
                    elements[i].rect = { x, y, x + 200, y + 20 };  // Define clickable area
                    SetTextColor(hdc, RGB(0, 0, 255));  // Links are typically blue
                    // Optionally underline the link
                    logfont.lfUnderline = TRUE;
                }

                WCHAR wText[1024];
                MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024);
                TextOutW(hdc, x, y, wText, wcslen(wText));

                DeleteObject(hFont);
                y += 30;
            }
        }

        EndPaint(hwnd, &ps);
    }
    break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_LBUTTONDOWN:
    {
        int xPos = GET_X_LPARAM(lParam);
        int yPos = GET_Y_LPARAM(lParam);

        for (int i = 0; i < element_count; i++) {
            if (elements[i].isLink && PtInRect(&elements[i].rect, { xPos, yPos })) {
                // Handle link click - for example, simulate navigation
                MessageBoxA(hwnd, elements[i].href, "Clicked Link", MB_OK);
            }
        }
    }
    break;

    case WM_SETCURSOR:
    {
        POINT pt;
        GetCursorPos(&pt);
        ScreenToClient(hwnd, &pt);

        for (int i = 0; i < element_count; i++) {
            if (elements[i].isLink && PtInRect(&elements[i].rect, pt)) {
                SetCursor(LoadCursor(NULL, IDC_HAND));
                return TRUE;  // We handled the cursor change
            }
        }
        SetCursor(LoadCursor(NULL, IDC_ARROW));
        return TRUE;
    }


    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Register the window class
    const char CLASS_NAME[] = "Sample Window Class";

    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCWSTR)CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles.
        (LPCWSTR)CLASS_NAME,                     // Window class
        L"Simple Browser",               // Window text
        WS_OVERLAPPEDWINDOW,            // Window style

        // Position and size
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL)
    {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    // Run the message loop
    MSG msg = { };
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif

#if 0
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    COLORREF textColor;       // Text color
    COLORREF backgroundColor; // Background color
    int fontSize;             // Font size
} CssStyle;

typedef struct {
    char text[1024];
    int bold;
    int italic;
    CssStyle style;
} HtmlElement;

// Convert a color string (like "#FF0000" for red) to COLORREF
COLORREF parse_color(const char* color) {
    if (color[0] == '#') {
        int r, g, b;
        sscanf(color, "#%02x%02x%02x", &r, &g, &b);
        return RGB(r, g, b);
    }
    // Default to black if no color specified
    return RGB(0, 0, 0);
}

// Very simple CSS parser for inline styles
void parse_css(const char* css, CssStyle* style) {
    const char* p = css;

    style->textColor = RGB(0, 0, 0);       // Default text color: black
    style->backgroundColor = RGB(255, 255, 255); // Default background: white
    style->fontSize = 20; // Default font size

    while (*p != '\0') {
        if (strncmp(p, "color:", 6) == 0) {
            p += 6;
            while (*p == ' ') p++;  // Skip spaces
            const char* color_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, color_start, p - color_start);
            color[p - color_start] = '\0';
            style->textColor = parse_color(color);
        }
        else if (strncmp(p, "background-color:", 17) == 0) {
            p += 17;
            while (*p == ' ') p++;  // Skip spaces
            const char* bg_start = p;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
            char color[8];
            strncpy(color, bg_start, p - bg_start);
            color[p - bg_start] = '\0';
            style->backgroundColor = parse_color(color);
        }
        else if (strncmp(p, "font-size:", 10) == 0) {
            p += 10;
            while (*p == ' ') p++;  // Skip spaces
            int size = 0;
            sscanf(p, "%d", &size); // Parse font size
            style->fontSize = size;
            while (*p != ';' && *p != '\0') p++; // Find the end of the value
        }
        while (*p != ';' && *p != '\0') p++;  // Skip to next style attribute
        if (*p == ';') p++;
    }
}

// Modify the parser to handle new tags and CSS styles
void parse_html(const char* html, HtmlElement* elements, int* element_count) {
    const char* p = html;
    *element_count = 0;

    while (*p != '\0') {
        if (*p == '<') {
            if (strncmp(p, "<p>", 3) == 0) {
                p += 3;  // Skip <p> tag
                continue;
            }
            else if (strncmp(p, "</p>", 4) == 0) {
                p += 4;
                continue;
            }
            else if (strncmp(p, "<div", 4) == 0) {
                elements[*element_count].bold = 0;
                elements[*element_count].italic = 0;
                p += 4;

                // Check if the <div> has a style attribute
                const char* style_start = strstr(p, "style=\"");
                if (style_start) {
                    style_start += 7; // Skip 'style="'
                    const char* style_end = strchr(style_start, '"');
                    char css[256];
                    strncpy(css, style_start, style_end - style_start);
                    css[style_end - style_start] = '\0';
                    parse_css(css, &elements[*element_count].style);
                    p = style_end + 1;
                }
                continue;
            }
            else if (strncmp(p, "</div>", 6) == 0) {
                p += 6;
                continue;
            }
        }
        else {
            // Read plain text
            char* text_start = (char*)p;
            while (*p != '<' && *p != '\0') {
                p++;
            }

            strncpy(elements[*element_count].text, text_start, p - text_start);
            elements[*element_count].text[p - text_start] = '\0';
            (*element_count)++;
        }
    }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    static HtmlElement elements[100];
    static int element_count = 0;

    const char* html_content = "<div style=\"color:#FF0000; background-color:#FFFF00; font-size:30;\">This is a styled div</div><div style=\"color:#0000FF; font-size:20;\">Another styled text</div>";

    switch (uMsg)
    {
    case WM_CREATE:
    {
        parse_html(html_content, elements, &element_count);
    }
    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        // Set initial drawing position
        int x = 10;
        int y = 10;

        for (int i = 0; i < element_count; i++) {
            // Set font based on bold and italic attributes
            HFONT hFont = NULL;
            LOGFONT logfont = { 0 };

            logfont.lfHeight = elements[i].style.fontSize;
            if (elements[i].bold) {
                logfont.lfWeight = FW_BOLD;
            }
            if (elements[i].italic) {
                logfont.lfItalic = TRUE;
            }

            hFont = CreateFontIndirect(&logfont);
            SelectObject(hdc, hFont);

            // Set text color
            SetTextColor(hdc, elements[i].style.textColor);

            // Set background color and fill the area
            RECT rect = { x, y, x + 800, y + elements[i].style.fontSize + 10 };
            HBRUSH hBrush = CreateSolidBrush(elements[i].style.backgroundColor);
            FillRect(hdc, &rect, hBrush);
            DeleteObject(hBrush);

            // Draw the text
            WCHAR wText[1024];
            MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024);
            TextOutW(hdc, x, y, wText, wcslen(wText));

            // Clean up
            DeleteObject(hFont);

            // Move to the next line
            y += elements[i].style.fontSize + 20;
        }

        EndPaint(hwnd, &ps);
    }
    break;


    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Register the window class
    const char CLASS_NAME[] = "Sample Window Class";

    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCWSTR)CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles.
        (LPCWSTR)CLASS_NAME,                     // Window class
        L"Simple Browser",               // Window text
        WS_OVERLAPPEDWINDOW,            // Window style

        // Position and size
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL)
    {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    // Run the message loop
    MSG msg = { };
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif


#if 0
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <tchar.h>


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow);

//void parse_html(const char* html, HtmlElement* elements, int* element_count);


// Struct to store parsed HTML content
typedef struct {
    char text[1024];
    int bold;
    int italic;
} HtmlElement;

void parse_html(const char* html, HtmlElement* elements, int* element_count) {
    const char* p = html;
    *element_count = 0;

    while (*p != '\0') {
        if (*p == '<') {
            if (strncmp(p, "<p>", 3) == 0) {
                p += 3;  // Skip the <p> tag
                continue;
            }
            else if (strncmp(p, "</p>", 4) == 0) {
                p += 4;
                continue;
            }
            else if (strncmp(p, "<b>", 3) == 0) {
                elements[*element_count].bold = 1;
                p += 3;
                continue;
            }
            else if (strncmp(p, "</b>", 4) == 0) {
                elements[*element_count].bold = 0;
                p += 4;
                continue;
            }
            else if (strncmp(p, "<i>", 3) == 0) {
                elements[*element_count].italic = 1;
                p += 3;
                continue;
            }
            else if (strncmp(p, "</i>", 4) == 0) {
                elements[*element_count].italic = 0;
                p += 4;
                continue;
            }
        }
        else {
            // Read plain text
            char* text_start = (char*)p;
            while (*p != '<' && *p != '\0') {
                p++;
            }

            strncpy(elements[*element_count].text, text_start, p - text_start);
            elements[*element_count].text[p - text_start] = '\0';
            (*element_count)++;
        }
    }
}


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    static HtmlElement elements[100];
    static int element_count = 0;

    const char* html_content = "<p>This is a <b>simple</b> HTML <i>parser</i> demo</p>";

    switch (uMsg)
    {
    case WM_CREATE:
    {
        parse_html(html_content, elements, &element_count);
    }
    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        // Set initial drawing position
        int x = 10;
        int y = 10;

        for (int i = 0; i < element_count; i++) {
            // Set font based on bold and italic attributes
            HFONT hFont = NULL;
            LOGFONT logfont = { 0 };

            logfont.lfHeight = 20;
            if (elements[i].bold) {
                logfont.lfWeight = FW_BOLD;
            }
            if (elements[i].italic) {
                logfont.lfItalic = TRUE;
            }

            hFont = CreateFontIndirect(&logfont);
            SelectObject(hdc, hFont);

            // Draw the text
            //TextOut(hdc, x, y, elements[i].text, strlen(elements[i].text));

            WCHAR wText[1024]; // Buffer for the wide-character string
            MultiByteToWideChar(CP_ACP, 0, elements[i].text, -1, wText, 1024); // Convert char* to wide char

            TextOutW(hdc, x, y, wText, wcslen(wText)); // Use TextOutW for wide strings


            // Clean up
            DeleteObject(hFont);

            // Move to the next line
            y += 30;
        }

        EndPaint(hwnd, &ps);
    }
    break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Register the window class
    const char CLASS_NAME[] = "Sample Window Class";

    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCWSTR)CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles.
        (LPCWSTR)CLASS_NAME,                     // Window class
        L"Simple Browser",               // Window text
        WS_OVERLAPPEDWINDOW,            // Window style

        // Position and size
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL)
    {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    // Run the message loop
    MSG msg = { };
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif



#if 0
// web_renderer.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "web_renderer.h"

#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Place code here.

    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_WEBRENDERER, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_WEBRENDERER));

    MSG msg;

    // Main message loop:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_WEBRENDERER));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_WEBRENDERER);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Store instance handle in our global variable

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // Parse the menu selections:
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code that uses hdc here...
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
#endif
