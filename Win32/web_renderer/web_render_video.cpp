#if 1
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
    vp.Width = 200.0f;
    vp.Height = 200.0f;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 10;  // Adjust as per your layout
    vp.TopLeftY = 10;
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
        SetTimer(hwnd, 1, 16, NULL);

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
        RECT videoRect = { 10, 10, 210, 210 }; // Adjust according to your video position
        InvalidateRect(hwnd, &videoRect, FALSE);
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
    SetTimer(hwnd, 1, 16, NULL); // 16ms timer for 60 FPS


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



