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
