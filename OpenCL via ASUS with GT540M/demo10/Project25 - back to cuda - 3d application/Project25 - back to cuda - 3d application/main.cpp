
#if 1
// cuda_gdi_demo_autoadd.cpp
// Single-file Win32 + GDI demo with CUDA probing (optional).
// Adds object count display and automatic +100 objects/second mode.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <thread>
#include <random>
#include <iostream>

// ---------------- Dynamic CUDA function typedefs ----------------
typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;
typedef CUresult(__stdcall* cuInit_t)(unsigned int);
typedef CUresult(__stdcall* cuDeviceGetAttribute_t)(int* pi, CUdevice_attribute attrib, CUdevice dev);

typedef int(__stdcall* cudaMalloc_t)(void**, size_t);
typedef int(__stdcall* cudaFree_t)(void*);
typedef int(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, int);
typedef int(__stdcall* cudaEventCreate_t)(void**);
typedef int(__stdcall* cudaEventRecord_t)(void*, void*);
typedef int(__stdcall* cudaEventSynchronize_t)(void*);
typedef int(__stdcall* cudaEventElapsedTime_t)(float*, void*, void*);
typedef int(__stdcall* cudaEventDestroy_t)(void*);
typedef int(__stdcall* cudaHostAlloc_t)(void**, size_t, unsigned int);
typedef int(__stdcall* cudaFreeHost_t)(void*);
typedef int(__stdcall* cudaMemGetInfo_t)(size_t*, size_t*);
typedef const char* (__stdcall* cudaGetErrorString_t)(int);

// cudaMemcpyKind values
enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// ---------------- Globals for CUDA pointers ----------------
static cuInit_t my_cuInit = nullptr;
static cuDeviceGetAttribute_t my_cuDeviceGetAttribute = nullptr;

static cudaMalloc_t my_cudaMalloc = nullptr;
static cudaFree_t my_cudaFree = nullptr;
static cudaMemcpy_t my_cudaMemcpy = nullptr;
static cudaEventCreate_t my_cudaEventCreate = nullptr;
static cudaEventRecord_t my_cudaEventRecord = nullptr;
static cudaEventSynchronize_t my_cudaEventSynchronize = nullptr;
static cudaEventElapsedTime_t my_cudaEventElapsedTime = nullptr;
static cudaEventDestroy_t my_cudaEventDestroy = nullptr;
static cudaHostAlloc_t my_cudaHostAlloc = nullptr;
static cudaFreeHost_t my_cudaFreeHost = nullptr;
static cudaMemGetInfo_t my_cudaMemGetInfo = nullptr;
static cudaGetErrorString_t my_cudaGetErrorString = nullptr;

// ---------------- Utility helpers ----------------
static void appendLog(const std::string& s) {
    const char* path = "C:\\Windows\\Temp\\log.txt";
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) return;
    ofs << s << "\n";
}

static std::wstring to_wstring_utf8(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n == 0) return std::wstring(s.begin(), s.end());
    std::wstring w; w.resize(n - 1);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

static std::string format_double(double v, int prec = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ---------------- GDI double-buffered rendering ----------------
struct GLibBuffer {
    HDC memDC = NULL;
    HBITMAP bitmap = NULL;
    int w = 0, h = 0;
    void ensure(HDC hdc, int width, int height) {
        if (width <= 0 || height <= 0) return;
        if (memDC && bitmap && width == w && height == h) return;
        if (memDC) { DeleteDC(memDC); memDC = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        memDC = CreateCompatibleDC(hdc);
        bitmap = CreateCompatibleBitmap(hdc, width, height);
        SelectObject(memDC, bitmap);
        w = width; h = height;
    }
    void release() {
        if (memDC) { DeleteDC(memDC); memDC = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        w = h = 0;
    }
};

// ---------------- Scene objects ----------------
struct Obj {
    float angle;
    float speed;
    POINT center;
    float size;
    COLORREF color;
};

static std::vector<Obj> g_objects; // scene objects (triangles)
static std::mt19937 g_rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
static const size_t MAX_OBJECTS = 200000; // safety cap

// Adds an object with random parameters centered randomly in the window
static void add_object(int winW, int winH) {
    if (g_objects.size() >= MAX_OBJECTS) return;
    std::uniform_real_distribution<float> ang(0.0f, 360.0f);
    std::uniform_real_distribution<float> spd(-2.0f, 2.0f);
    std::uniform_real_distribution<float> cx(0.1f * winW, 0.9f * winW);
    std::uniform_real_distribution<float> cy(0.1f * winH, 0.9f * winH);
    std::uniform_real_distribution<float> sz(10.0f, min(winW, winH) * 0.12f);
    std::uniform_int_distribution<int> col(40, 255);
    Obj o;
    o.angle = ang(g_rng);
    o.speed = spd(g_rng);
    o.center.x = LONG(cx(g_rng));
    o.center.y = LONG(cy(g_rng));
    o.size = sz(g_rng);
    o.color = RGB(col(g_rng), col(g_rng), col(g_rng));
    g_objects.push_back(o);
}

static void add_n_objects(int winW, int winH, int n) {
    for (int i = 0; i < n; ++i) {
        if (g_objects.size() >= MAX_OBJECTS) break;
        add_object(winW, winH);
    }
}

static void reset_objects() {
    g_objects.clear();
}

// draw filled triangle given center/size/angle and color
static void draw_triangle_instance(HDC mem, int w, int h, const Obj& o) {
    float cx = (float)o.center.x;
    float cy = (float)o.center.y;
    float size = o.size;
    float a = o.angle * 3.14159265f / 180.0f;
    float px[3] = { 0.0f, -0.6f, 0.6f };
    float py[3] = { -1.0f, 0.4f, 0.4f };
    POINT pts[3];
    for (int i = 0; i < 3; i++) {
        float x = px[i] * size;
        float y = py[i] * size;
        float xr = x * cosf(a) - y * sinf(a);
        float yr = x * sinf(a) + y * cosf(a);
        pts[i].x = LONG(cx + xr);
        pts[i].y = LONG(cy + yr);
    }

    HBRUSH brush = CreateSolidBrush(o.color);
    HBRUSH oldBrush = (HBRUSH)SelectObject(mem, brush);
    HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
    HPEN oldPen = (HPEN)SelectObject(mem, pen);
    Polygon(mem, pts, 3);
    SelectObject(mem, oldBrush); DeleteObject(brush);
    SelectObject(mem, oldPen); DeleteObject(pen);
}

// draw background
static void clear_bg(HDC mem, int w, int h) {
    HBRUSH bg = CreateSolidBrush(RGB(15, 15, 30));
    RECT r = { 0,0,w,h };
    FillRect(mem, &r, bg);
    DeleteObject(bg);
}

// overlay: show SM count, object count, autoadd status
static void draw_info_box(HDC mem, int w, int h, int smCount, size_t objCount, bool autoAdd) {
    SetBkMode(mem, TRANSPARENT);
    SetTextColor(mem, RGB(220, 220, 220));
    HFONT fnt = CreateFontW(16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 10, 10, 320, 120 };
    std::wostringstream ws;
    if (smCount >= 0) ws << L"SMs: " << smCount << L"\n";
    else ws << L"SMs: n/a\n";
    ws << L"Objects: " << objCount << L"\n";
    ws << L"AutoAdd: " << (autoAdd ? L"ON" : L"OFF");
    DrawTextW(mem, ws.str().c_str(), -1, &rc, DT_LEFT | DT_TOP | DT_NOPREFIX);
    SelectObject(mem, old);
    DeleteObject(fnt);
}

// draw FPS numeric big in center of screen
static void draw_fps_big(HDC mem, int w, int h, double fps) {
    std::wostringstream ws;
    ws << L"FPS: " << std::fixed << std::setprecision(1) << fps;
    std::wstring s = ws.str();
    SetBkMode(mem, TRANSPARENT);
    SetTextColor(mem, RGB(240, 240, 40));
    HFONT fnt = CreateFontW(44, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 0, 0, w, h };
    DrawTextW(mem, s.c_str(), -1, &rc, DT_CENTER | DT_VCENTER | DT_NOPREFIX);
    SelectObject(mem, old);
    DeleteObject(fnt);
}

// FPS graph (lower-right). fpsHistory latest at end
static void draw_fps_graph(HDC mem, int w, int h, const std::vector<double>& fpsHistory) {
    const int graphW = 240;
    const int graphH = 90;
    const int pad = 8;
    int gx = w - graphW - pad;
    int gy = h - graphH - pad;
    RECT bgrect = { gx, gy, gx + graphW, gy + graphH };
    HBRUSH bg = CreateSolidBrush(RGB(20, 20, 40));
    FillRect(mem, &bgrect, bg);
    DeleteObject(bg);

    // border
    HPEN pen = CreatePen(PS_SOLID, 1, RGB(120, 120, 160));
    HPEN oldPen = (HPEN)SelectObject(mem, pen);
    SelectObject(mem, GetStockObject(NULL_BRUSH));
    Rectangle(mem, gx, gy, gx + graphW, gy + graphH);

    // compute scale: y range 0..maxFPS (auto max with min 30)
    double maxFPS = 60.0;
    for (double v : fpsHistory) if (v > maxFPS) maxFPS = v;
    if (maxFPS < 30.0) maxFPS = 30.0;

    // draw polyline
    int n = (int)fpsHistory.size();
    if (n >= 2) {
        std::vector<POINT> pts;
        pts.reserve(n);
        int start = max(0, n - (graphW - 12));
        int samples = n - start;
        for (int i = start; i < n; ++i) {
            double v = fpsHistory[i];
            double norm = v / maxFPS;
            if (norm < 0) norm = 0;
            if (norm > 1) norm = 1;
            int x = gx + 6 + (int)((double)(i - start) * (graphW - 12) / double(max(1, samples - 1)));
            int y = gy + graphH - 6 - (int)(norm * (graphH - 12));
            pts.push_back({ x,y });
        }
        HPEN gpen = CreatePen(PS_SOLID, 2, RGB(40, 240, 40));
        HPEN oldG = (HPEN)SelectObject(mem, gpen);
        for (size_t i = 1; i < pts.size(); ++i) {
            MoveToEx(mem, pts[i - 1].x, pts[i - 1].y, NULL);
            LineTo(mem, pts[i].x, pts[i].y);
        }
        SelectObject(mem, oldG);
        DeleteObject(gpen);
    }

    // draw axis labels small
    SetTextColor(mem, RGB(180, 180, 220));
    HFONT small = CreateFontW(12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT oldF = (HFONT)SelectObject(mem, small);
    std::wstring labelHigh = to_wstring_utf8(format_double(maxFPS, 0));
    TextOutW(mem, gx + 6, gy + 4, labelHigh.c_str(), (int)labelHigh.size());
    std::wstring labelLow = L"0";
    TextOutW(mem, gx + 6, gy + graphH - 18, labelLow.c_str(), (int)labelLow.size());
    SelectObject(mem, oldF);
    DeleteObject(small);

    SelectObject(mem, oldPen);
    DeleteObject(pen);
}

// ---------------- Menu command IDs ----------------
enum {
    IDM_ADD_OBJECT = 1001,
    IDM_RESET_OBJECTS = 1002,
    IDM_TOGGLE_AUTOADD = 1003,
    IDM_ADD_100_NOW = 1004
};

// ---------------- Globals controlling UI state ----------------
static int g_smCount = -1;                 // from driver API (or -1 if n/a)
static bool g_cudaAvailable = false;
static std::vector<double> g_fpsHistory;   // accumulate history
static const size_t HISTORY_MAX = 1024;
static bool g_autoAdd = false;

// ---------------- Window Proc ----------------
LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDM_ADD_OBJECT:
        {
            RECT r; GetClientRect(hwnd, &r);
            add_object(r.right - r.left, r.bottom - r.top);
        }
        return 0;
        case IDM_ADD_100_NOW:
        {
            RECT r; GetClientRect(hwnd, &r);
            add_n_objects(r.right - r.left, r.bottom - r.top, 100);
        }
        return 0;
        case IDM_RESET_OBJECTS:
            reset_objects();
            return 0;
        case IDM_TOGGLE_AUTOADD:
            g_autoAdd = !g_autoAdd;
            return 0;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// ---------------- Main ----------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
    // register class
    const wchar_t* cls = L"CUDA_GDI_MenuGraph_AutoAdd";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.style = CS_OWNDC;
    RegisterClassW(&wc);

    // create window
    HWND hwnd = CreateWindowExW(0, cls, L"CUDA GDI Demo (auto-add + graph)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1000, 720, NULL, NULL, hInst, NULL);
    if (!hwnd) return 1;
    HDC hdc = GetDC(hwnd);

    // create menu
    HMENU hMenu = CreateMenu();
    HMENU hSub = CreatePopupMenu();
    AppendMenuW(hSub, MF_STRING, IDM_ADD_OBJECT, L"Add Object");
    AppendMenuW(hSub, MF_STRING, IDM_ADD_100_NOW, L"Add 100 Now");
    AppendMenuW(hSub, MF_STRING, IDM_RESET_OBJECTS, L"Reset Objects");
    AppendMenuW(hSub, MF_SEPARATOR, 0, NULL);
    AppendMenuW(hSub, MF_STRING, IDM_TOGGLE_AUTOADD, L"Toggle AutoAdd (100/s)");
    AppendMenuW(hMenu, MF_POPUP, (UINT_PTR)hSub, L"Objects");
    SetMenu(hwnd, hMenu);

    // load driver (nvcuda.dll)
    HMODULE hDriver = LoadLibraryW(L"nvcuda.dll");
    if (hDriver) {
        my_cuInit = (cuInit_t)GetProcAddress(hDriver, "cuInit");
        my_cuDeviceGetAttribute = (cuDeviceGetAttribute_t)GetProcAddress(hDriver, "cuDeviceGetAttribute");
        if (!my_cuInit || !my_cuDeviceGetAttribute) { my_cuInit = nullptr; my_cuDeviceGetAttribute = nullptr; }
        else {
            CUresult r = my_cuInit(0);
            if (r == 0) {
                int val = 0;
                if (my_cuDeviceGetAttribute(&val, 16, 0) == 0) { g_smCount = val; g_cudaAvailable = true; }
            }
        }
    }

    // load runtime (cudart.dll) optional
    HMODULE hRuntime = LoadLibraryW(L"C:\\CUDA\\bin\\cudart.dll");
    if (!hRuntime) hRuntime = LoadLibraryW(L"cudart.dll");
    if (hRuntime) {
        my_cudaMalloc = (cudaMalloc_t)GetProcAddress(hRuntime, "cudaMalloc");
        my_cudaFree = (cudaFree_t)GetProcAddress(hRuntime, "cudaFree");
        my_cudaMemcpy = (cudaMemcpy_t)GetProcAddress(hRuntime, "cudaMemcpy");
        my_cudaEventCreate = (cudaEventCreate_t)GetProcAddress(hRuntime, "cudaEventCreate");
        my_cudaEventRecord = (cudaEventRecord_t)GetProcAddress(hRuntime, "cudaEventRecord");
        my_cudaEventSynchronize = (cudaEventSynchronize_t)GetProcAddress(hRuntime, "cudaEventSynchronize");
        my_cudaEventElapsedTime = (cudaEventElapsedTime_t)GetProcAddress(hRuntime, "cudaEventElapsedTime");
        my_cudaEventDestroy = (cudaEventDestroy_t)GetProcAddress(hRuntime, "cudaEventDestroy");
        my_cudaHostAlloc = (cudaHostAlloc_t)GetProcAddress(hRuntime, "cudaHostAlloc");
        my_cudaFreeHost = (cudaFreeHost_t)GetProcAddress(hRuntime, "cudaFreeHost");
        my_cudaMemGetInfo = (cudaMemGetInfo_t)GetProcAddress(hRuntime, "cudaMemGetInfo");
        my_cudaGetErrorString = (cudaGetErrorString_t)GetProcAddress(hRuntime, "cudaGetErrorString");
    }

    // prepare 4MB host buffer (try pinned)
    const size_t TRANSFER_BYTES = 4 * 1024 * 1024;
    void* hostBuf = nullptr; bool hostPinned = false;
    if (my_cudaHostAlloc) {
        if (my_cudaHostAlloc(&hostBuf, TRANSFER_BYTES, 0) == 0 && hostBuf) hostPinned = true;
        else hostBuf = nullptr;
    }
    if (!hostBuf) {
        hostBuf = malloc(TRANSFER_BYTES);
    }
    if (hostBuf) memset(hostBuf, 0xA5, TRANSFER_BYTES);

    // device buffer
    void* devBuf = nullptr; bool haveDevBuf = false;
    if (my_cudaMalloc) {
        if (my_cudaMalloc(&devBuf, TRANSFER_BYTES) == 0 && devBuf) haveDevBuf = true;
    }

    // events
    void* evStart = nullptr; void* evStop = nullptr; bool haveEvents = false;
    if (my_cudaEventCreate && my_cudaEventRecord && my_cudaEventElapsedTime && my_cudaEventSynchronize && my_cudaEventDestroy) {
        if (my_cudaEventCreate(&evStart) == 0 && my_cudaEventCreate(&evStop) == 0) haveEvents = true;
        else { if (evStart) my_cudaEventDestroy(evStart); if (evStop) my_cudaEventDestroy(evStop); evStart = evStop = nullptr; }
    }

    // initial objects (one triangle)
    RECT rc; GetClientRect(hwnd, &rc);
    add_object(rc.right - rc.left, rc.bottom - rc.top);

    // GDI buffer
    GLibBuffer gbuf;
    GetClientRect(hwnd, &rc);
    gbuf.ensure(hdc, rc.right - rc.left, rc.bottom - rc.top);

    // timing / FPS
    using clock = std::chrono::steady_clock;
    auto lastSecond = clock::now();
    auto lastFrame = clock::now();
    int frames = 0;
    double displayFPS = 0.0;
    double lastH2D_ms = -1.0, lastD2H_ms = -1.0;
    size_t lastFree = 0, lastTotal = 0;

    double autoAddAccumulator = 0.0; // accumulate dt for auto-add
    bool running = true;
    MSG msg;
    while (running) {
        // message pump
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) { running = false; break; }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;

        // refresh buf if needed
        GetClientRect(hwnd, &rc);
        gbuf.ensure(hdc, rc.right - rc.left, rc.bottom - rc.top);

        // update objects (angles)
        auto nowFrame = clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(nowFrame - lastFrame).count();
        if (dt <= 0) dt = 0.016f;
        lastFrame = nowFrame;
        for (auto& o : g_objects) o.angle += o.speed * dt * 60.0f;

        // auto-add logic: add 100 objects each second if enabled
        if (g_autoAdd) {
            autoAddAccumulator += dt;
            while (autoAddAccumulator >= 1.0) {
                add_n_objects(gbuf.w, gbuf.h, 100);
                autoAddAccumulator -= 1.0;
            }
        }
        else {
            autoAddAccumulator = 0.0;
        }

        // draw background & objects
        clear_bg(gbuf.memDC, gbuf.w, gbuf.h);
        for (const auto& o : g_objects) draw_triangle_instance(gbuf.memDC, gbuf.w, gbuf.h, o);

        // draw info box (SMs + objects + autoadd status)
        draw_info_box(gbuf.memDC, gbuf.w, gbuf.h, g_smCount, g_objects.size(), g_autoAdd);

        // draw FPS big on animation screen (center)
        draw_fps_big(gbuf.memDC, gbuf.w, gbuf.h, displayFPS);

        // draw FPS graph in lower-right
        draw_fps_graph(gbuf.memDC, gbuf.w, gbuf.h, g_fpsHistory);

        // blit
        BitBlt(hdc, 0, 0, gbuf.w, gbuf.h, gbuf.memDC, 0, 0, SRCCOPY);

        // CUDA transfer measurements (if available)
        if (haveDevBuf && my_cudaMemcpy) {
            if (haveEvents) {
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                float ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastH2D_ms = ms;
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastD2H_ms = ms;
            }
            else {
                auto t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                auto t1 = std::chrono::high_resolution_clock::now();
                lastH2D_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
                t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                t1 = std::chrono::high_resolution_clock::now();
                lastD2H_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
            }
        }

        // mem info
        if (my_cudaMemGetInfo) {
            size_t freeB = 0, totalB = 0;
            if (my_cudaMemGetInfo(&freeB, &totalB) == 0) { lastFree = freeB; lastTotal = totalB; }
        }

        // frames and per-second updates
        frames++;
        auto now = clock::now();
        double dtsec = std::chrono::duration_cast<std::chrono::duration<double>>(now - lastSecond).count();
        if (dtsec >= 0.25) { // update graph more frequently for smoother plot (quarter-second)
            displayFPS = double(frames) / dtsec;
            // push history
            g_fpsHistory.push_back(displayFPS);
            if (g_fpsHistory.size() > HISTORY_MAX) g_fpsHistory.erase(g_fpsHistory.begin(), g_fpsHistory.begin() + (g_fpsHistory.size() - HISTORY_MAX));
            // update window title small (optional)
            std::ostringstream title;
            title << "CUDA GDI Demo - Objects=" << g_objects.size() << " | SMs=" << (g_smCount >= 0 ? std::to_string(g_smCount) : "n/a") << (g_autoAdd ? " | AutoAdd=ON" : "");
            SetWindowTextW(hwnd, to_wstring_utf8(title.str()).c_str());

            // log every second only
            static double accumLog = 0.0;
            accumLog += dtsec;
            if (accumLog >= 1.0) {
                std::ostringstream log;
                log << "FPS=" << std::fixed << std::setprecision(1) << displayFPS
                    << " Objects=" << g_objects.size()
                    << " SMs=" << (g_smCount >= 0 ? std::to_string(g_smCount) : "n/a")
                    << " freeMB=" << (lastTotal > 0 ? std::to_string(lastFree / (1024ULL * 1024ULL)) : "n/a");
                appendLog(log.str());
                accumLog = 0.0;
            }

            frames = 0;
            lastSecond = now;
        }

        // tiny sleep to yield CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
    }

    // cleanup
    gbuf.release();
    if (evStart && my_cudaEventDestroy) my_cudaEventDestroy(evStart);
    if (evStop && my_cudaEventDestroy) my_cudaEventDestroy(evStop);
    if (devBuf && my_cudaFree) my_cudaFree(devBuf);
    if (hostBuf) {
        if (hostPinned && my_cudaFreeHost) my_cudaFreeHost(hostBuf);
        else free(hostBuf);
    }
    if (hRuntime) FreeLibrary(hRuntime);
    if (hDriver) FreeLibrary(hDriver);
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    return 0;
}
#endif


#if 0
// cuda_gdi_demo_menu_graph.cpp
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <thread>
#include <random>
#include <iostream>

// ---------------- Dynamic CUDA function typedefs ----------------
typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;
typedef CUresult(__stdcall* cuInit_t)(unsigned int);
typedef CUresult(__stdcall* cuDeviceGetAttribute_t)(int* pi, CUdevice_attribute attrib, CUdevice dev);

typedef int(__stdcall* cudaMalloc_t)(void**, size_t);
typedef int(__stdcall* cudaFree_t)(void*);
typedef int(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, int);
typedef int(__stdcall* cudaEventCreate_t)(void**);
typedef int(__stdcall* cudaEventRecord_t)(void*, void*);
typedef int(__stdcall* cudaEventSynchronize_t)(void*);
typedef int(__stdcall* cudaEventElapsedTime_t)(float*, void*, void*);
typedef int(__stdcall* cudaEventDestroy_t)(void*);
typedef int(__stdcall* cudaHostAlloc_t)(void**, size_t, unsigned int);
typedef int(__stdcall* cudaFreeHost_t)(void*);
typedef int(__stdcall* cudaMemGetInfo_t)(size_t*, size_t*);
typedef const char* (__stdcall* cudaGetErrorString_t)(int);

// cudaMemcpyKind values
enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// ---------------- Globals for CUDA pointers ----------------
static cuInit_t my_cuInit = nullptr;
static cuDeviceGetAttribute_t my_cuDeviceGetAttribute = nullptr;

static cudaMalloc_t my_cudaMalloc = nullptr;
static cudaFree_t my_cudaFree = nullptr;
static cudaMemcpy_t my_cudaMemcpy = nullptr;
static cudaEventCreate_t my_cudaEventCreate = nullptr;
static cudaEventRecord_t my_cudaEventRecord = nullptr;
static cudaEventSynchronize_t my_cudaEventSynchronize = nullptr;
static cudaEventElapsedTime_t my_cudaEventElapsedTime = nullptr;
static cudaEventDestroy_t my_cudaEventDestroy = nullptr;
static cudaHostAlloc_t my_cudaHostAlloc = nullptr;
static cudaFreeHost_t my_cudaFreeHost = nullptr;
static cudaMemGetInfo_t my_cudaMemGetInfo = nullptr;
static cudaGetErrorString_t my_cudaGetErrorString = nullptr;

// ---------------- Utility helpers ----------------
static void appendLog(const std::string& s) {
    const char* path = "C:\\Windows\\Temp\\log.txt";
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) return;
    ofs << s << "\n";
}

static std::wstring to_wstring_utf8(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n == 0) return std::wstring(s.begin(), s.end());
    std::wstring w; w.resize(n - 1);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

static std::string format_double(double v, int prec = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ---------------- GDI double-buffered rendering ----------------
struct GLibBuffer {
    HDC memDC = NULL;
    HBITMAP bitmap = NULL;
    int w = 0, h = 0;
    void ensure(HDC hdc, int width, int height) {
        if (width <= 0 || height <= 0) return;
        if (memDC && bitmap && width == w && height == h) return;
        if (memDC) { DeleteDC(memDC); memDC = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        memDC = CreateCompatibleDC(hdc);
        bitmap = CreateCompatibleBitmap(hdc, width, height);
        SelectObject(memDC, bitmap);
        w = width; h = height;
    }
    void release() {
        if (memDC) { DeleteDC(memDC); memDC = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        w = h = 0;
    }
};

// ---------------- Scene objects ----------------
struct Obj {
    float angle;
    float speed;
    POINT center;
    float size;
    COLORREF color;
};

static std::vector<Obj> g_objects; // scene objects (triangles)
static std::mt19937 g_rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());

// Adds an object with random parameters centered randomly in the window
static void add_object(int winW, int winH) {
    std::uniform_real_distribution<float> ang(0.0f, 360.0f);
    std::uniform_real_distribution<float> spd(-2.0f, 2.0f);
    std::uniform_real_distribution<float> cx(0.2f * winW, 0.8f * winW);
    std::uniform_real_distribution<float> cy(0.2f * winH, 0.8f * winH);
    std::uniform_real_distribution<float> sz(20.0f, min(winW, winH) * 0.15f);
    std::uniform_int_distribution<int> col(60, 255);
    Obj o;
    o.angle = ang(g_rng);
    o.speed = spd(g_rng);
    o.center.x = LONG(cx(g_rng));
    o.center.y = LONG(cy(g_rng));
    o.size = sz(g_rng);
    o.color = RGB(col(g_rng), col(g_rng), col(g_rng));
    g_objects.push_back(o);
}

static void reset_objects() {
    g_objects.clear();
}

// draw filled triangle given center/size/angle and color
static void draw_triangle_instance(HDC mem, int w, int h, const Obj& o) {
    float cx = (float)o.center.x;
    float cy = (float)o.center.y;
    float size = o.size;
    float a = o.angle * 3.14159265f / 180.0f;
    float px[3] = { 0.0f, -0.6f, 0.6f };
    float py[3] = { -1.0f, 0.4f, 0.4f };
    POINT pts[3];
    for (int i = 0; i < 3; i++) {
        float x = px[i] * size;
        float y = py[i] * size;
        float xr = x * cosf(a) - y * sinf(a);
        float yr = x * sinf(a) + y * cosf(a);
        pts[i].x = LONG(cx + xr);
        pts[i].y = LONG(cy + yr);
    }

    HBRUSH brush = CreateSolidBrush(o.color);
    HBRUSH oldBrush = (HBRUSH)SelectObject(mem, brush);
    HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
    HPEN oldPen = (HPEN)SelectObject(mem, pen);
    Polygon(mem, pts, 3);
    SelectObject(mem, oldBrush); DeleteObject(brush);
    SelectObject(mem, oldPen); DeleteObject(pen);
}

// draw background
static void clear_bg(HDC mem, int w, int h) {
    HBRUSH bg = CreateSolidBrush(RGB(15, 15, 30));
    RECT r = { 0,0,w,h };
    FillRect(mem, &r, bg);
    DeleteObject(bg);
}

// overlay: only show SM count; everything else "N/A"
static void draw_info_box(HDC mem, int w, int h, int smCount) {
    SetBkMode(mem, TRANSPARENT);
    SetTextColor(mem, RGB(220, 220, 220));
    HFONT fnt = CreateFontW(18, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 10, 10, 250, 90 };
    std::wostringstream ws;
    if (smCount >= 0) ws << L"SMs: " << smCount << L"\n";
    else ws << L"SMs: n/a\n";
    // per your request, only the SM line is shown — the rest we show as N/A succinctly
    ws << L"Other: N/A";
    DrawTextW(mem, ws.str().c_str(), -1, &rc, DT_LEFT | DT_TOP | DT_NOPREFIX);
    SelectObject(mem, old);
    DeleteObject(fnt);
}

// draw FPS numeric big in center of screen
static void draw_fps_big(HDC mem, int w, int h, double fps) {
    std::wostringstream ws;
    ws << L"FPS: " << std::fixed << std::setprecision(1) << fps;
    std::wstring s = ws.str();
    SetBkMode(mem, TRANSPARENT);
    SetTextColor(mem, RGB(240, 240, 40));
    HFONT fnt = CreateFontW(48, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 0, 0, w, h };
    DrawTextW(mem, s.c_str(), -1, &rc, DT_CENTER | DT_VCENTER | DT_NOPREFIX);
    SelectObject(mem, old);
    DeleteObject(fnt);
}

// FPS graph (lower-right). fpsHistory latest at end
static void draw_fps_graph(HDC mem, int w, int h, const std::vector<double>& fpsHistory) {
    const int graphW = 240;
    const int graphH = 90;
    const int pad = 8;
    int gx = w - graphW - pad;
    int gy = h - graphH - pad;
    RECT bgrect = { gx, gy, gx + graphW, gy + graphH };
    HBRUSH bg = CreateSolidBrush(RGB(20, 20, 40));
    FillRect(mem, &bgrect, bg);
    DeleteObject(bg);

    // border
    HPEN pen = CreatePen(PS_SOLID, 1, RGB(120, 120, 160));
    HPEN oldPen = (HPEN)SelectObject(mem, pen);
    SelectObject(mem, GetStockObject(NULL_BRUSH));
    Rectangle(mem, gx, gy, gx + graphW, gy + graphH);

    // compute scale: y range 0..maxFPS (auto max with min 30)
    double maxFPS = 60.0;
    for (double v : fpsHistory) if (v > maxFPS) maxFPS = v;
    if (maxFPS < 30.0) maxFPS = 30.0;

    // draw polyline
    int n = (int)fpsHistory.size();
    if (n >= 2) {
        std::vector<POINT> pts;
        pts.reserve(n);
        int start = max(0, n - (graphW - 10)); // limit samples to graph width
        int samples = n - start;
        for (int i = start; i < n; ++i) {
            double v = fpsHistory[i];
            double norm = v / maxFPS;
            if (norm < 0) norm = 0;
            if (norm > 1) norm = 1;
            int x = gx + 6 + (int)((double)(i - start) * (graphW - 12) / double(max(1, samples - 1)));
            int y = gy + graphH - 6 - (int)(norm * (graphH - 12));
            pts.push_back({ x,y });
        }
        // draw lines
        HPEN gpen = CreatePen(PS_SOLID, 2, RGB(40, 240, 40));
        HPEN oldG = (HPEN)SelectObject(mem, gpen);
        for (size_t i = 1; i < pts.size(); ++i) {
            MoveToEx(mem, pts[i - 1].x, pts[i - 1].y, NULL);
            LineTo(mem, pts[i].x, pts[i].y);
        }
        SelectObject(mem, oldG);
        DeleteObject(gpen);
    }

    // draw axis labels small
    SetTextColor(mem, RGB(180, 180, 220));
    HFONT small = CreateFontW(12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT oldF = (HFONT)SelectObject(mem, small);
    std::wstring labelHigh = std::wstring(L"") + std::wstring(L"max ") + std::wstring(to_wstring_utf8(format_double(maxFPS, 0)));
    TextOutW(mem, gx + 6, gy + 4, labelHigh.c_str(), (int)labelHigh.size());
    std::wstring labelLow = std::wstring(L"0");
    TextOutW(mem, gx + 6, gy + graphH - 18, labelLow.c_str(), (int)labelLow.size());
    SelectObject(mem, oldF);
    DeleteObject(small);

    SelectObject(mem, oldPen);
    DeleteObject(pen);
}

// ---------------- Menu command IDs ----------------
enum {
    IDM_ADD_OBJECT = 1001,
    IDM_RESET_OBJECTS = 1002
};

// ---------------- Globals controlling UI state ----------------
static int g_smCount = -1;                 // from driver API (or -1 if n/a)
static bool g_cudaAvailable = false;
static std::vector<double> g_fpsHistory;   // accumulate history
static const size_t HISTORY_MAX = 512;

// ---------------- Window Proc (we need WM_COMMAND handling for menu) ----------------
LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDM_ADD_OBJECT:
        {
            RECT r; GetClientRect(hwnd, &r);
            add_object(r.right - r.left, r.bottom - r.top);
        }
        return 0;
        case IDM_RESET_OBJECTS:
            reset_objects();
            return 0;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// ---------------- Main ----------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
    // register class
    const wchar_t* cls = L"CUDA_GDI_MenuGraph";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.style = CS_OWNDC;
    RegisterClassW(&wc);

    // create window
    HWND hwnd = CreateWindowExW(0, cls, L"CUDA GDI Demo (menu + FPS graph)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1000, 720, NULL, NULL, hInst, NULL);
    if (!hwnd) return 1;
    HDC hdc = GetDC(hwnd);

    // create menu
    HMENU hMenu = CreateMenu();
    HMENU hSub = CreatePopupMenu();
    AppendMenuW(hSub, MF_STRING, IDM_ADD_OBJECT, L"Add Object");
    AppendMenuW(hSub, MF_STRING, IDM_RESET_OBJECTS, L"Reset Objects");
    AppendMenuW(hMenu, MF_POPUP, (UINT_PTR)hSub, L"Objects");
    SetMenu(hwnd, hMenu);

    // load driver (nvcuda.dll)
    HMODULE hDriver = LoadLibraryW(L"nvcuda.dll");
    if (hDriver) {
        my_cuInit = (cuInit_t)GetProcAddress(hDriver, "cuInit");
        my_cuDeviceGetAttribute = (cuDeviceGetAttribute_t)GetProcAddress(hDriver, "cuDeviceGetAttribute");
        if (!my_cuInit || !my_cuDeviceGetAttribute) { my_cuInit = nullptr; my_cuDeviceGetAttribute = nullptr; }
        else {
            CUresult r = my_cuInit(0);
            if (r == 0) {
                int val = 0;
                if (my_cuDeviceGetAttribute(&val, 16, 0) == 0) { g_smCount = val; g_cudaAvailable = true; }
            }
        }
    }

    // load runtime (cudart.dll) optional
    HMODULE hRuntime = LoadLibraryW(L"C:\\CUDA\\bin\\cudart.dll");
    if (!hRuntime) hRuntime = LoadLibraryW(L"cudart.dll");
    if (hRuntime) {
        my_cudaMalloc = (cudaMalloc_t)GetProcAddress(hRuntime, "cudaMalloc");
        my_cudaFree = (cudaFree_t)GetProcAddress(hRuntime, "cudaFree");
        my_cudaMemcpy = (cudaMemcpy_t)GetProcAddress(hRuntime, "cudaMemcpy");
        my_cudaEventCreate = (cudaEventCreate_t)GetProcAddress(hRuntime, "cudaEventCreate");
        my_cudaEventRecord = (cudaEventRecord_t)GetProcAddress(hRuntime, "cudaEventRecord");
        my_cudaEventSynchronize = (cudaEventSynchronize_t)GetProcAddress(hRuntime, "cudaEventSynchronize");
        my_cudaEventElapsedTime = (cudaEventElapsedTime_t)GetProcAddress(hRuntime, "cudaEventElapsedTime");
        my_cudaEventDestroy = (cudaEventDestroy_t)GetProcAddress(hRuntime, "cudaEventDestroy");
        my_cudaHostAlloc = (cudaHostAlloc_t)GetProcAddress(hRuntime, "cudaHostAlloc");
        my_cudaFreeHost = (cudaFreeHost_t)GetProcAddress(hRuntime, "cudaFreeHost");
        my_cudaMemGetInfo = (cudaMemGetInfo_t)GetProcAddress(hRuntime, "cudaMemGetInfo");
        my_cudaGetErrorString = (cudaGetErrorString_t)GetProcAddress(hRuntime, "cudaGetErrorString");
    }

    // prepare 4MB host buffer (try pinned)
    const size_t TRANSFER_BYTES = 4 * 1024 * 1024;
    void* hostBuf = nullptr; bool hostPinned = false;
    if (my_cudaHostAlloc) {
        if (my_cudaHostAlloc(&hostBuf, TRANSFER_BYTES, 0) == 0 && hostBuf) hostPinned = true;
        else hostBuf = nullptr;
    }
    if (!hostBuf) {
        hostBuf = malloc(TRANSFER_BYTES);
    }
    if (hostBuf) memset(hostBuf, 0xA5, TRANSFER_BYTES);

    // device buffer
    void* devBuf = nullptr; bool haveDevBuf = false;
    if (my_cudaMalloc) {
        if (my_cudaMalloc(&devBuf, TRANSFER_BYTES) == 0 && devBuf) haveDevBuf = true;
    }

    // events
    void* evStart = nullptr; void* evStop = nullptr; bool haveEvents = false;
    if (my_cudaEventCreate && my_cudaEventRecord && my_cudaEventElapsedTime && my_cudaEventSynchronize && my_cudaEventDestroy) {
        if (my_cudaEventCreate(&evStart) == 0 && my_cudaEventCreate(&evStop) == 0) haveEvents = true;
        else { if (evStart) my_cudaEventDestroy(evStart); if (evStop) my_cudaEventDestroy(evStop); evStart = evStop = nullptr; }
    }

    // initial objects (one triangle)
    RECT rc; GetClientRect(hwnd, &rc);
    add_object(rc.right - rc.left, rc.bottom - rc.top);

    // GDI buffer
    GLibBuffer gbuf;
    GetClientRect(hwnd, &rc);
    gbuf.ensure(hdc, rc.right - rc.left, rc.bottom - rc.top);

    // timing / FPS
    using clock = std::chrono::steady_clock;
    auto lastSecond = clock::now();
    auto lastFrame = clock::now();
    int frames = 0;
    double displayFPS = 0.0;
    double lastH2D_ms = -1.0, lastD2H_ms = -1.0;
    size_t lastFree = 0, lastTotal = 0;

    bool running = true;
    MSG msg;
    while (running) {
        // message pump
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) { running = false; break; }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;

        // refresh buf if needed
        GetClientRect(hwnd, &rc);
        gbuf.ensure(hdc, rc.right - rc.left, rc.bottom - rc.top);

        // update objects (angles)
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(clock::now() - lastFrame).count();
        if (dt <= 0) dt = 0.016f;
        lastFrame = clock::now();
        for (auto& o : g_objects) o.angle += o.speed * dt * 60.0f; // scale by fps-ish
        // also keep main object list maybe large

        // draw background & objects
        clear_bg(gbuf.memDC, gbuf.w, gbuf.h);
        for (const auto& o : g_objects) draw_triangle_instance(gbuf.memDC, gbuf.w, gbuf.h, o);

        // draw info box (only SMs)
        draw_info_box(gbuf.memDC, gbuf.w, gbuf.h, g_smCount);

        // draw FPS big on animation screen (center)
        draw_fps_big(gbuf.memDC, gbuf.w, gbuf.h, displayFPS);

        // draw FPS graph in lower-right
        draw_fps_graph(gbuf.memDC, gbuf.w, gbuf.h, g_fpsHistory);

        // blit
        BitBlt(hdc, 0, 0, gbuf.w, gbuf.h, gbuf.memDC, 0, 0, SRCCOPY);

        // CUDA transfer measurements (if available)
        if (haveDevBuf && my_cudaMemcpy) {
            if (haveEvents) {
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                float ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastH2D_ms = ms;
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastD2H_ms = ms;
            }
            else {
                auto t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                auto t1 = std::chrono::high_resolution_clock::now();
                lastH2D_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
                t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                t1 = std::chrono::high_resolution_clock::now();
                lastD2H_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
            }
        }

        // mem info
        if (my_cudaMemGetInfo) {
            size_t freeB = 0, totalB = 0;
            if (my_cudaMemGetInfo(&freeB, &totalB) == 0) { lastFree = freeB; lastTotal = totalB; }
        }

        // frames and per-second updates
        frames++;
        auto now = clock::now();
        double dtsec = std::chrono::duration_cast<std::chrono::duration<double>>(now - lastSecond).count();
        if (dtsec >= 0.25) { // update graph more frequently for smoother plot (quarter-second)
            displayFPS = double(frames) / dtsec;
            // push history
            g_fpsHistory.push_back(displayFPS);
            if (g_fpsHistory.size() > HISTORY_MAX) g_fpsHistory.erase(g_fpsHistory.begin(), g_fpsHistory.begin() + (g_fpsHistory.size() - HISTORY_MAX));
            // update window title small (optional)
            std::ostringstream title;
            title << "CUDA GDI Demo - Objects=" << g_objects.size() << " | SMs=" << (g_smCount >= 0 ? std::to_string(g_smCount) : "n/a");
            SetWindowTextW(hwnd, to_wstring_utf8(title.str()).c_str());

            // log every second only
            static double accumLog = 0.0;
            accumLog += dtsec;
            if (accumLog >= 1.0) {
                std::ostringstream log;
                log << "FPS=" << std::fixed << std::setprecision(1) << displayFPS
                    << " Objects=" << g_objects.size()
                    << " SMs=" << (g_smCount >= 0 ? std::to_string(g_smCount) : "n/a")
                    << " freeMB=" << (lastTotal > 0 ? std::to_string(lastFree / (1024ULL * 1024ULL)) : "n/a");
                appendLog(log.str());
                accumLog = 0.0;
            }

            frames = 0;
            lastSecond = now;
        }

        // tiny sleep so the UI yields
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
    }

    // cleanup
    gbuf.release();
    if (evStart && my_cudaEventDestroy) my_cudaEventDestroy(evStart);
    if (evStop && my_cudaEventDestroy) my_cudaEventDestroy(evStop);
    if (devBuf && my_cudaFree) my_cudaFree(devBuf);
    if (hostBuf) {
        if (hostPinned && my_cudaFreeHost) my_cudaFreeHost(hostBuf);
        else free(hostBuf);
    }
    if (hRuntime) FreeLibrary(hRuntime);
    if (hDriver) FreeLibrary(hDriver);
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    return 0;
}
#endif


#if 0
// cuda_gdi_demo_fixed.cpp
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <thread>
#include <iostream>

// ---------------- Dynamic CUDA function typedefs ----------------
typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;
typedef CUresult(__stdcall* cuInit_t)(unsigned int);
typedef CUresult(__stdcall* cuDeviceGetAttribute_t)(int* pi, CUdevice_attribute attrib, CUdevice dev);

typedef int(__stdcall* cudaMalloc_t)(void**, size_t);
typedef int(__stdcall* cudaFree_t)(void*);
typedef int(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, int);
typedef int(__stdcall* cudaEventCreate_t)(void**);
typedef int(__stdcall* cudaEventRecord_t)(void*, void*);
typedef int(__stdcall* cudaEventSynchronize_t)(void*);
typedef int(__stdcall* cudaEventElapsedTime_t)(float*, void*, void*);
typedef int(__stdcall* cudaEventDestroy_t)(void*);
typedef int(__stdcall* cudaHostAlloc_t)(void**, size_t, unsigned int);
typedef int(__stdcall* cudaFreeHost_t)(void*);
typedef int(__stdcall* cudaMemGetInfo_t)(size_t*, size_t*);
typedef const char* (__stdcall* cudaGetErrorString_t)(int);

// cudaMemcpyKind values
enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// ---------------- Globals ----------------
static cuInit_t my_cuInit = nullptr;
static cuDeviceGetAttribute_t my_cuDeviceGetAttribute = nullptr;

static cudaMalloc_t my_cudaMalloc = nullptr;
static cudaFree_t my_cudaFree = nullptr;
static cudaMemcpy_t my_cudaMemcpy = nullptr;
static cudaEventCreate_t my_cudaEventCreate = nullptr;
static cudaEventRecord_t my_cudaEventRecord = nullptr;
static cudaEventSynchronize_t my_cudaEventSynchronize = nullptr;
static cudaEventElapsedTime_t my_cudaEventElapsedTime = nullptr;
static cudaEventDestroy_t my_cudaEventDestroy = nullptr;
static cudaHostAlloc_t my_cudaHostAlloc = nullptr;
static cudaFreeHost_t my_cudaFreeHost = nullptr;
static cudaMemGetInfo_t my_cudaMemGetInfo = nullptr;
static cudaGetErrorString_t my_cudaGetErrorString = nullptr;

// ---------------- Utility helpers ----------------
static void appendLog(const std::string& s) {
    const char* path = "C:\\Windows\\Temp\\log.txt";
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) return;
    ofs << s << "\n";
}

static std::wstring to_wstring_utf8(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n == 0) return std::wstring(s.begin(), s.end());
    std::wstring w; w.resize(n - 1);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

static std::string format_double(double v, int prec = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ---------------- GDI double-buffered rendering ----------------
struct GLibBuffer {
    HDC memDC = NULL;
    HBITMAP bitmap = NULL;
    int w = 0, h = 0;
    void ensure(HDC hdc, int width, int height) {
        if (width <= 0 || height <= 0) return;
        if (memDC && bitmap && width == w && height == h) return;
        if (memDC) { DeleteDC(memDC); memDC = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        memDC = CreateCompatibleDC(hdc);
        bitmap = CreateCompatibleBitmap(hdc, width, height);
        SelectObject(memDC, bitmap);
        w = width; h = height;
    }
    void release() {
        if (memDC) { DeleteDC(memDC); memDC = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        w = h = 0;
    }
};

static void draw_triangle(HDC mem, int w, int h, float angleDeg) {
    HBRUSH bg = CreateSolidBrush(RGB(15, 15, 30));
    RECT r = { 0,0,w,h };
    FillRect(mem, &r, bg);
    DeleteObject(bg);

    float cx = w * 0.5f, cy = h * 0.45f;
    float size = min(w, h) * 0.4f;
    float a = angleDeg * 3.14159265f / 180.0f;
    float px[3] = { 0.0f, -0.6f, 0.6f };
    float py[3] = { -1.0f, 0.4f, 0.4f };
    POINT pts[3];
    for (int i = 0; i < 3; i++) {
        float x = px[i] * size;
        float y = py[i] * size;
        float xr = x * cosf(a) - y * sinf(a);
        float yr = x * sinf(a) + y * cosf(a);
        pts[i].x = LONG(cx + xr);
        pts[i].y = LONG(cy + yr);
    }

    HBRUSH brush = CreateSolidBrush(RGB(200, 60, 60));
    HBRUSH oldBrush = (HBRUSH)SelectObject(mem, brush);
    HPEN pen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
    HPEN oldPen = (HPEN)SelectObject(mem, pen);
    Polygon(mem, pts, 3);
    SelectObject(mem, oldBrush); DeleteObject(brush);
    SelectObject(mem, oldPen); DeleteObject(pen);
}

static void draw_stats(HDC mem, int w, int h, const std::wstring& lines) {
    SetBkMode(mem, TRANSPARENT);
    SetTextColor(mem, RGB(230, 230, 230));
    HFONT fnt = CreateFontW(18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 10, 10, w - 10, h - 10 };
    DrawTextW(mem, lines.c_str(), -1, &rc, DT_LEFT | DT_TOP | DT_NOPREFIX);
    SelectObject(mem, old);
    DeleteObject(fnt);
}

// ---------------- Main application ----------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
    const wchar_t* cls = L"CUDA_GDI_Demo";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.style = CS_OWNDC;
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(0, cls, L"CUDA GDI Demo", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 900, 700, NULL, NULL, hInst, NULL);
    if (!hwnd) return 1;

    HDC hdc = GetDC(hwnd);

    // load driver (nvcuda.dll)
    HMODULE hDriver = LoadLibraryW(L"nvcuda.dll");
    if (hDriver) {
        my_cuInit = (cuInit_t)GetProcAddress(hDriver, "cuInit");
        my_cuDeviceGetAttribute = (cuDeviceGetAttribute_t)GetProcAddress(hDriver, "cuDeviceGetAttribute");
        if (!my_cuInit || !my_cuDeviceGetAttribute) { my_cuInit = nullptr; my_cuDeviceGetAttribute = nullptr; }
        else {
            CUresult r = my_cuInit(0);
            if (r != 0) { my_cuInit = nullptr; my_cuDeviceGetAttribute = nullptr; }
        }
    }

    // load runtime (cudart.dll)
    HMODULE hRuntime = LoadLibraryW(L"C:\\CUDA\\bin\\cudart.dll");
    if (!hRuntime) hRuntime = LoadLibraryW(L"cudart.dll");
    if (hRuntime) {
        my_cudaMalloc = (cudaMalloc_t)GetProcAddress(hRuntime, "cudaMalloc");
        my_cudaFree = (cudaFree_t)GetProcAddress(hRuntime, "cudaFree");
        my_cudaMemcpy = (cudaMemcpy_t)GetProcAddress(hRuntime, "cudaMemcpy");
        my_cudaEventCreate = (cudaEventCreate_t)GetProcAddress(hRuntime, "cudaEventCreate");
        my_cudaEventRecord = (cudaEventRecord_t)GetProcAddress(hRuntime, "cudaEventRecord");
        my_cudaEventSynchronize = (cudaEventSynchronize_t)GetProcAddress(hRuntime, "cudaEventSynchronize");
        my_cudaEventElapsedTime = (cudaEventElapsedTime_t)GetProcAddress(hRuntime, "cudaEventElapsedTime");
        my_cudaEventDestroy = (cudaEventDestroy_t)GetProcAddress(hRuntime, "cudaEventDestroy");
        my_cudaHostAlloc = (cudaHostAlloc_t)GetProcAddress(hRuntime, "cudaHostAlloc");
        my_cudaFreeHost = (cudaFreeHost_t)GetProcAddress(hRuntime, "cudaFreeHost");
        my_cudaMemGetInfo = (cudaMemGetInfo_t)GetProcAddress(hRuntime, "cudaMemGetInfo");
        my_cudaGetErrorString = (cudaGetErrorString_t)GetProcAddress(hRuntime, "cudaGetErrorString");
    }

    // prepare 4MB host buffer (try pinned)
    const size_t TRANSFER_BYTES = 4 * 1024 * 1024;
    void* hostBuf = nullptr; bool hostPinned = false;
    if (my_cudaHostAlloc) {
        if (my_cudaHostAlloc(&hostBuf, TRANSFER_BYTES, 0) == 0 && hostBuf) hostPinned = true;
        else hostBuf = nullptr;
    }
    if (!hostBuf) {
        hostBuf = malloc(TRANSFER_BYTES);
    }
    if (hostBuf) memset(hostBuf, 0xA5, TRANSFER_BYTES);

    // device buffer
    void* devBuf = nullptr; bool haveDevBuf = false;
    if (my_cudaMalloc) {
        if (my_cudaMalloc(&devBuf, TRANSFER_BYTES) == 0 && devBuf) haveDevBuf = true;
    }

    // events
    void* evStart = nullptr; void* evStop = nullptr; bool haveEvents = false;
    if (my_cudaEventCreate && my_cudaEventRecord && my_cudaEventElapsedTime && my_cudaEventSynchronize && my_cudaEventDestroy) {
        if (my_cudaEventCreate(&evStart) == 0 && my_cudaEventCreate(&evStop) == 0) haveEvents = true;
        else { if (evStart) my_cudaEventDestroy(evStart); if (evStop) my_cudaEventDestroy(evStop); evStart = evStop = nullptr; }
    }

    // driver GPU SM count
    int smCount = -1;
    if (my_cuDeviceGetAttribute) {
        int val = 0;
        if (my_cuDeviceGetAttribute(&val, 16, 0) == 0) smCount = val; // 16 is MULTIPROCESSOR_COUNT
    }

    // GDI buffer
    GLibBuffer gbuf;
    RECT rRect; GetClientRect(hwnd, &rRect);
    gbuf.ensure(hdc, rRect.right - rRect.left, rRect.bottom - rRect.top);

    // main loop timing
    using clock = std::chrono::steady_clock;
    auto lastSecond = clock::now();
    int frames = 0;
    float angle = 0.0f;

    double lastH2D_ms = -1.0, lastD2H_ms = -1.0;
    size_t lastFree = 0, lastTotal = 0;

    bool running = true;
    MSG msg;
    while (running) {
        // message pump
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        // ensure buffer size
        GetClientRect(hwnd, &rRect);
        gbuf.ensure(hdc, rRect.right - rRect.left, rRect.bottom - rRect.top);

        // update simulation
        angle += 0.9f;
        if (angle >= 360.0f) angle -= 360.0f;

        // draw into memDC
        draw_triangle(gbuf.memDC, gbuf.w, gbuf.h, angle);

        // stats string
        std::ostringstream ss;
        ss << "CUDA GDI Demo\n";
        ss << "SMs: " << (smCount >= 0 ? std::to_string(smCount) : std::string("n/a")) << "\n";
        ss << "H2D(ms): " << (lastH2D_ms >= 0.0 ? format_double(lastH2D_ms, 3) : std::string("n/a")) << "\n";
        ss << "D2H(ms): " << (lastD2H_ms >= 0.0 ? format_double(lastD2H_ms, 3) : std::string("n/a")) << "\n";
        if (lastTotal > 0) ss << "GPU free MB: " << (lastFree / (1024ULL * 1024ULL)) << "/" << (lastTotal / (1024ULL * 1024ULL)) << "\n";
        else ss << "GPU mem: n/a\n";

        std::wstring wlines = to_wstring_utf8(ss.str());
        draw_stats(gbuf.memDC, gbuf.w, gbuf.h, wlines);

        // blit to screen
        BitBlt(hdc, 0, 0, gbuf.w, gbuf.h, gbuf.memDC, 0, 0, SRCCOPY);

        // perform CUDA transfer measurements (if available)
        if (haveDevBuf && my_cudaMemcpy) {
            if (haveEvents) {
                // H2D
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                float ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastH2D_ms = ms;
                // D2H
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastD2H_ms = ms;
            }
            else {
                auto t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                auto t1 = std::chrono::high_resolution_clock::now();
                lastH2D_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
                t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                t1 = std::chrono::high_resolution_clock::now();
                lastD2H_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
            }
        }

        // mem info
        if (my_cudaMemGetInfo) {
            size_t freeB = 0, totalB = 0;
            if (my_cudaMemGetInfo(&freeB, &totalB) == 0) { lastFree = freeB; lastTotal = totalB; }
        }

        // FPS + logging per second
        frames++;
        auto now = clock::now();
        double dtsec = std::chrono::duration_cast<std::chrono::duration<double>>(now - lastSecond).count();
        if (dtsec >= 1.0) {
            double fps = double(frames) / dtsec;
            std::ostringstream title;
            title << "CUDA GDI Demo - FPS=" << std::fixed << std::setprecision(1) << fps;
            if (lastTotal > 0) title << " freeMB=" << (lastFree / (1024ULL * 1024ULL));
            if (smCount >= 0) title << " SMs=" << smCount;
            std::wstring wtitle = to_wstring_utf8(title.str());
            SetWindowTextW(hwnd, wtitle.c_str());

            std::ostringstream log;
            log << "FPS=" << std::fixed << std::setprecision(1) << fps
                << " H2D_ms=" << (lastH2D_ms >= 0 ? format_double(lastH2D_ms, 3) : std::string("n/a"))
                << " D2H_ms=" << (lastD2H_ms >= 0 ? format_double(lastD2H_ms, 3) : std::string("n/a"))
                << " SMs=" << (smCount >= 0 ? std::to_string(smCount) : std::string("n/a"))
                << " freeMB=" << (lastTotal > 0 ? std::to_string(lastFree / (1024ULL * 1024ULL)) : "n/a");
            appendLog(log.str());

            frames = 0;
            lastSecond = now;
        }

        // small sleep to yield CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }

    // cleanup
    gbuf.release();
    if (evStart && my_cudaEventDestroy) my_cudaEventDestroy(evStart);
    if (evStop && my_cudaEventDestroy) my_cudaEventDestroy(evStop);
    if (haveDevBuf && my_cudaFree) my_cudaFree(devBuf);
    if (hostBuf) {
        if (hostPinned && my_cudaFreeHost) my_cudaFreeHost(hostBuf);
        else free(hostBuf);
    }
    if (hRuntime) FreeLibrary(hRuntime);
    if (hDriver) FreeLibrary(hDriver);
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    return 0;
}
#endif
