#if 1 
// ---------------------------
// Win32 main + loop that uses gpu_build_polygons_driver_api (or CPU fallback)
// Paste this AFTER the gpu_build_polygons_driver_api function in your file.
// ---------------------------

// ======= Paste THIS at the very top of main.cpp (immediately after #includes) =======
// Core types, helpers and forward declarations so the rest of the file compiles.

#include <windows.h>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <algorithm>

// ---- small utility helpers (same as earlier) ----
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

// ---- object / polygon types used across file ----
enum ObjType {
    OBJ_TRIANGLE = 0,
    OBJ_CYLINDER = 1,
    OBJ_BOX = 2
};

struct Obj {
    ObjType type = OBJ_TRIANGLE;
    // legacy 2D fields (kept for backward compatibility)
    POINT center = { 0,0 };
    float angle = 0.0f;
    float speed = 0.0f;

    // 3D fields
    float cx = 0.0f;
    float cy = 0.0f;
    float z = 0.0f;
    float size = 16.0f;
    float rotX = 0.0f;
    float rotY = 0.0f;
    float rotZ = 0.0f;
    float rotSpeedX = 0.0f;
    float rotSpeedY = 0.0f;
    float rotSpeedZ = 0.0f;
    int cylSegments = 16;
    float thickness = 8.0f;
    COLORREF color = RGB(200, 200, 200);
    Obj() = default;
};

struct DrawPoly {
    std::vector<POINT> pts;
    float depth = 0.0f;
    COLORREF color = RGB(200, 200, 200);
};

// ---- globals used across file (declared here) ----
static std::vector<Obj> g_objects;          // declared early so other code can reference it
static std::mt19937 g_rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());

// Minimal stub implementation so the project links.
// It intentionally returns false to indicate GPU path not available;
// the caller will fall back to CPU polygon builder.
bool gpu_build_polygons_driver_api(
    const std::vector<Obj>& objs,
    int screenW, int screenH,
    float camYaw, float camPitch, float camDistance,
    std::vector<DrawPoly>& outPolys,
    double* gpuMsOut /*= nullptr*/)
{
    // Explicitly silence unused parameter warnings
    (void)objs; (void)screenW; (void)screenH; (void)camYaw; (void)camPitch; (void)camDistance;
    // indicate no GPU work done
    if (gpuMsOut) *gpuMsOut = 0.0;
    // return false -> caller will do CPU fallback
    return false;
}


// ---- forward declarations for functions implemented later in the file ----
// GPU/driver loader wrapper (implemented previously in the file)
bool gpu_build_polygons_driver_api(
    const std::vector<Obj>& objs,
    int screenW, int screenH,
    float camYaw, float camPitch, float camDistance,
    std::vector<DrawPoly>& outPolys,
    double* gpuMsOut = nullptr);

// CPU fallback polygon builder (implemented later)
void build_obj_polys_cpu(const Obj& o, int w, int h, std::vector<DrawPoly>& out);

// GDI draw helper (implemented later)
void draw_polys_gdi(HDC mem, const std::vector<DrawPoly>& polys);

// UI/drawing helpers (implemented later)
void clear_bg(HDC mem, int w, int h);
void draw_info_box(HDC mem, int w, int h, bool gpuSuccess, double gpuMs);
void draw_fps_big(HDC mem, int w, int h, double fps);
void draw_fps_graph(HDC mem, int w, int h, const std::vector<double>& fpsHistory);

// main entry forward (defined later by the Win32 loop you asked for)
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR pCmdLine, int nCmdShow);


// Globals for scene & UI
//static std::vector<Obj> g_objects;
static std::vector<double> g_fpsHistory;
static bool g_useGpuIfAvailable = true;
static float g_camYaw = 0.0f, g_camPitch = 0.0f, g_camDistance = 600.0f;
static bool g_mouseDown = false;
static POINT g_mouseStart = { 0,0 };
static float g_startYaw = 0.0f, g_startPitch = 0.0f;
//static std::mt19937 g_rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());



// Forward declarations
static void clear_bg(HDC mem, int w, int h);
static void draw_fps_big(HDC mem, int w, int h, double fps);
static void draw_fps_graph(HDC mem, int w, int h, const std::vector<double>& fpsHistory);
static void draw_info_box(HDC mem, int w, int h, bool gpuSuccess, double gpuMs);

// Simple projection used by CPU fallback (matches kernel's projection roughly)
static bool project_point_camera(int screenW, int screenH, float wx, float wy, float wz, POINT& out, float& outDepth) {
    float x = wx - screenW * 0.5f;
    float y = wy - screenH * 0.5f;
    float z = wz;
    // inverse camera rotations
    auto rotate_xyz = [](float& x, float& y, float& z, float ax, float ay, float az) {
        float rx = ax * 3.14159265358979323846f / 180.0f;
        float ry = ay * 3.14159265358979323846f / 180.0f;
        float rz = az * 3.14159265358979323846f / 180.0f;
        // X
        float cx = cosf(rx), sx = sinf(rx);
        float y1 = y * cx - z * sx;
        float z1 = y * sx + z * cx;
        y = y1; z = z1;
        // Y
        float cy = cosf(ry), sy = sinf(ry);
        float x1 = x * cy + z * sy;
        float z2 = -x * sy + z * cy;
        x = x1; z = z2;
        // Z
        float cz = cosf(rz), sz = sinf(rz);
        float x2 = x * cz - y * sz;
        float y2 = x * sz + y * cz;
        x = x2; y = y2;
    };
    rotate_xyz(x, y, z, -g_camPitch, -g_camYaw, 0.0f);
    float zcam = z + g_camDistance;
    if (zcam < 1e-3f) zcam = 1e-3f;
    float focal = (float)screenW * 0.8f;
    float sx = (x * (focal / zcam)) + screenW * 0.5f;
    float sy = (y * (focal / zcam)) + screenH * 0.5f;
    out.x = LONG(sx);
    out.y = LONG(sy);
    outDepth = zcam;
    return true;
}

// CPU fallback: build polygons for one Obj (triangles, boxes, cylinders simplified)
static void build_obj_polys_cpu(const Obj& o, int w, int h, std::vector<DrawPoly>& out) {
    if (o.type == OBJ_TRIANGLE) {
        // simple triangular plate
        float s = o.size;
        float lx[3] = { 0.0f, -0.6f * s, 0.6f * s };
        float ly[3] = { -1.0f * s, 0.4f * s, 0.4f * s };
        float lz[3] = { -o.thickness * 0.2f, o.thickness * 0.2f, 0.0f };
        POINT pts[3];
        float depths[3];
        for (int i = 0; i < 3; ++i) {
            float x = lx[i], y = ly[i], z = lz[i];
            // rotate by object rotation
            auto rotate = [](float& x, float& y, float& z, float ax, float ay, float az) {
                float rx = ax * 3.14159265358979323846f / 180.0f;
                float ry = ay * 3.14159265358979323846f / 180.0f;
                float rz = az * 3.14159265358979323846f / 180.0f;
                // X
                float cx = cosf(rx), sx = sinf(rx);
                float y1 = y * cx - z * sx;
                float z1 = y * sx + z * cx;
                y = y1; z = z1;
                // Y
                float cy = cosf(ry), sy = sinf(ry);
                float x1 = x * cy + z * sy;
                float z2 = -x * sy + z * cy;
                x = x1; z = z2;
                // Z
                float cz = cosf(rz), sz = sinf(rz);
                float x2 = x * cz - y * sz;
                float y2 = x * sz + y * cz;
                x = x2; y = y2;
            };
            rotate(x, y, z, o.rotX, o.rotY, o.rotZ);
            float wx = o.cx + x;
            float wy = o.cy + y;
            float wz = o.z + z;
            float dep;
            project_point_camera(w, h, wx, wy, wz, pts[i], dep);
            depths[i] = dep;
        }
        DrawPoly p;
        p.color = o.color;
        p.pts.assign(pts, pts + 3);
        p.depth = (depths[0] + depths[1] + depths[2]) / 3.0f;
        out.push_back(std::move(p));
    }
    else if (o.type == OBJ_BOX) {
        float hw = o.size * 0.5f, hh = o.size * 0.5f, hd = o.thickness * 0.5f;
        float vx[8] = { -hw, hw, hw, -hw, -hw, hw, hw, -hw };
        float vy[8] = { -hh, -hh, hh, hh, -hh, -hh, hh, hh };
        float vz[8] = { -hd, -hd, -hd, -hd, hd, hd, hd, hd };
        POINT pv[8]; float pd[8];
        for (int i = 0; i < 8; ++i) {
            float x = vx[i], y = vy[i], z = vz[i];
            // rotate
            auto rotate = [](float& x, float& y, float& z, float ax, float ay, float az) {
                float rx = ax * 3.14159265358979323846f / 180.0f;
                float ry = ay * 3.14159265358979323846f / 180.0f;
                float rz = az * 3.14159265358979323846f / 180.0f;
                // X
                float cx = cosf(rx), sx = sinf(rx);
                float y1 = y * cx - z * sx;
                float z1 = y * sx + z * cx;
                y = y1; z = z1;
                // Y
                float cy = cosf(ry), sy = sinf(ry);
                float x1 = x * cy + z * sy;
                float z2 = -x * sy + z * cy;
                x = x1; z = z2;
                // Z
                float cz = cosf(rz), sz = sinf(rz);
                float x2 = x * cz - y * sz;
                float y2 = x * sz + y * cz;
                x = x2; y = y2;
            };
            rotate(x, y, z, o.rotX, o.rotY, o.rotZ);
            float wx = o.cx + x, wy = o.cy + y, wz = o.z + z;
            project_point_camera(w, h, wx, wy, wz, pv[i], pd[i]);
        }
        const int faceIdx[6][4] = { {0,1,2,3},{4,5,6,7},{0,1,5,4},{2,3,7,6},{1,2,6,5},{3,0,4,7} };
        for (int f = 0; f < 6; ++f) {
            DrawPoly p; p.color = o.color;
            p.pts.resize(4);
            float avg = 0.0f;
            for (int k = 0; k < 4; ++k) {
                int idx = faceIdx[f][k];
                p.pts[k] = pv[idx];
                avg += pd[idx];
            }
            p.depth = avg / 4.0f;
            out.push_back(std::move(p));
        }
    }
    else if (o.type == OBJ_CYLINDER) {
        int segs = o.cylSegments;
        if (segs < 6) segs = 6;
        float radius = o.size * 0.6f;
        float halfh = o.thickness * 0.5f;
        std::vector<POINT> top(segs), bot(segs);
        std::vector<float> td(segs), bd(segs);
        for (int i = 0; i < segs; ++i) {
            float theta = i * 2.0f * 3.14159265358979323846f / segs;
            float lx = cosf(theta) * radius, ly = sinf(theta) * radius;
            float lzt = halfh, lzb = -halfh;
            float xt = lx, yt = ly, zt = lzt;
            float xb = lx, yb = ly, zb = lzb;
            auto rotate = [](float& x, float& y, float& z, float ax, float ay, float az) {
                float rx = ax * 3.14159265358979323846f / 180.0f;
                float ry = ay * 3.14159265358979323846f / 180.0f;
                float rz = az * 3.14159265358979323846f / 180.0f;
                // X
                float cx = cosf(rx), sx = sinf(rx);
                float y1 = y * cx - z * sx;
                float z1 = y * sx + z * cx;
                y = y1; z = z1;
                // Y
                float cy = cosf(ry), sy = sinf(ry);
                float x1 = x * cy + z * sy;
                float z2 = -x * sy + z * cy;
                x = x1; z = z2;
                // Z
                float cz = cosf(rz), sz = sinf(rz);
                float x2 = x * cz - y * sz;
                float y2 = x * sz + y * cz;
                x = x2; y = y2;
            };
            rotate(xt, yt, zt, o.rotX, o.rotY, o.rotZ);
            rotate(xb, yb, zb, o.rotX, o.rotY, o.rotZ);
            project_point_camera(w, h, o.cx + xt, o.cy + yt, o.z + zt, top[i], td[i]);
            project_point_camera(w, h, o.cx + xb, o.cy + yb, o.z + zb, bot[i], bd[i]);
        }
        // top face as fan
        DrawPoly ptop; ptop.color = o.color; ptop.pts = top; ptop.depth = 0;
        for (float v : td) ptop.depth += v; ptop.depth /= segs;
        out.push_back(std::move(ptop));
        // bottom face
        DrawPoly pbot; pbot.color = o.color; pbot.pts = bot; pbot.depth = 0;
        for (float v : bd) pbot.depth += v; pbot.depth /= segs;
        out.push_back(std::move(pbot));
        // sides
        for (int i = 0; i < segs; ++i) {
            int ni = (i + 1) % segs;
            DrawPoly ps; ps.color = o.color; ps.pts.resize(4);
            ps.pts[0] = top[i]; ps.pts[1] = top[ni]; ps.pts[2] = bot[ni]; ps.pts[3] = bot[i];
            ps.depth = (td[i] + td[ni] + bd[ni] + bd[i]) * 0.25f;
            out.push_back(std::move(ps));
        }
    }
}

// draw polygons (GDI)
static void draw_polys_gdi(HDC mem, const std::vector<DrawPoly>& polys) {
    for (const auto& p : polys) {
        if (p.pts.empty()) continue;
        HBRUSH br = CreateSolidBrush(p.color);
        HBRUSH oldB = (HBRUSH)SelectObject(mem, br);
        HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
        HPEN oldP = (HPEN)SelectObject(mem, pen);
        Polygon(mem, p.pts.data(), (int)p.pts.size());
        SelectObject(mem, oldB); DeleteObject(br);
        SelectObject(mem, oldP); DeleteObject(pen);
    }
}

// helper: add random object
static void add_random_obj(int w, int h, ObjType type) {
    std::uniform_real_distribution<float> cx(0.1f * w, 0.9f * w);
    std::uniform_real_distribution<float> cy(0.1f * h, 0.9f * h);
    std::uniform_real_distribution<float> sz(8.0f, min(w, h) * 0.12f);
    std::uniform_real_distribution<float> zpos(w * 0.2f, w * 1.2f);
    std::uniform_real_distribution<float> ang(0.0f, 360.0f);
    std::uniform_real_distribution<float> spd(-60.0f, 60.0f);
    std::uniform_int_distribution<int> seg(12, 28);
    std::uniform_int_distribution<int> col(40, 255);
    std::uniform_real_distribution<float> depth(6.0f, 80.0f);

    Obj o;
    o.type = type;
    o.cx = cx(g_rng);
    o.cy = cy(g_rng);
    o.z = zpos(g_rng);
    o.size = sz(g_rng);
    o.rotX = ang(g_rng);
    o.rotY = ang(g_rng);
    o.rotZ = ang(g_rng);
    o.rotSpeedX = spd(g_rng);
    o.rotSpeedY = spd(g_rng);
    o.rotSpeedZ = spd(g_rng);
    o.cylSegments = seg(g_rng);
    o.thickness = depth(g_rng);
    o.color = RGB(col(g_rng), col(g_rng), col(g_rng));
    g_objects.push_back(o);
}

// window proc
LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case 1001: { RECT r; GetClientRect(hwnd, &r); add_random_obj(r.right - r.left, r.bottom - r.top, OBJ_TRIANGLE); } return 0;
        case 1002: { RECT r; GetClientRect(hwnd, &r); add_random_obj(r.right - r.left, r.bottom - r.top, OBJ_CYLINDER); } return 0;
        case 1003: { RECT r; GetClientRect(hwnd, &r); add_random_obj(r.right - r.left, r.bottom - r.top, OBJ_BOX); } return 0;
        case 1004: g_objects.clear(); return 0;
        case 1005: g_useGpuIfAvailable = !g_useGpuIfAvailable; return 0;
        }
        break;
    case WM_LBUTTONDOWN:
        SetCapture(hwnd);
        g_mouseDown = true;
        g_mouseStart.x = LOWORD(lParam);
        g_mouseStart.y = HIWORD(lParam);
        g_startYaw = g_camYaw;
        g_startPitch = g_camPitch;
        return 0;
    case WM_LBUTTONUP:
        ReleaseCapture();
        g_mouseDown = false;
        return 0;
    case WM_MOUSEMOVE:
        if (g_mouseDown) {
            int mx = LOWORD(lParam), my = HIWORD(lParam);
            g_camYaw = g_startYaw + (mx - g_mouseStart.x) * 0.25f;
            g_camPitch = g_startPitch + (my - g_mouseStart.y) * 0.25f;
            if (g_camPitch > 89.0f) g_camPitch = 89.0f;
            if (g_camPitch < -89.0f) g_camPitch = -89.0f;
        }
        return 0;
    case WM_MOUSEWHEEL: {
        short delta = GET_WHEEL_DELTA_WPARAM(wParam);
        float factor = (delta > 0) ? 0.9f : 1.1f;
        int steps = abs(delta) / WHEEL_DELTA;
        for (int i = 0; i < steps; ++i) g_camDistance *= factor;
        if (g_camDistance < 50.0f) g_camDistance = 50.0f;
        if (g_camDistance > 5000.0f) g_camDistance = 5000.0f;
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// helper drawing implementations
static void clear_bg(HDC mem, int w, int h) {
    if (!mem || w <= 0 || h <= 0) return;
    RECT r = { 0,0,w,h };
    HBRUSH bg = CreateSolidBrush(RGB(15, 15, 30));
    FillRect(mem, &r, bg);
    DeleteObject(bg);
}
static void draw_fps_big(HDC mem, int w, int h, double fps) {
    std::wostringstream ws; ws << L"FPS: " << std::fixed << std::setprecision(1) << fps;
    SetBkMode(mem, TRANSPARENT); SetTextColor(mem, RGB(240, 240, 40));
    HFONT fnt = CreateFontW(44, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 0,0,w,h };
    DrawTextW(mem, ws.str().c_str(), -1, &rc, DT_CENTER | DT_VCENTER | DT_NOPREFIX);
    SelectObject(mem, old); DeleteObject(fnt);
}
static void draw_fps_graph(HDC mem, int w, int h, const std::vector<double>& fpsHistory) {
    // same as earlier simplified
    const int graphW = 240, graphH = 90, pad = 8;
    int gx = w - graphW - pad, gy = h - graphH - pad;
    RECT bgrect = { gx, gy, gx + graphW, gy + graphH };
    HBRUSH bg = CreateSolidBrush(RGB(20, 20, 40)); FillRect(mem, &bgrect, bg); DeleteObject(bg);
    HPEN pen = CreatePen(PS_SOLID, 1, RGB(120, 120, 160)); HPEN oldPen = (HPEN)SelectObject(mem, pen);
    Rectangle(mem, gx, gy, gx + graphW, gy + graphH);
    double maxFPS = 60.0; for (double v : fpsHistory) if (v > maxFPS) maxFPS = v; if (maxFPS < 30.0) maxFPS = 30.0;
    int n = (int)fpsHistory.size();
    if (n >= 2) {
        std::vector<POINT> pts; pts.reserve(n);
        int start = max(0, n - (graphW - 12)); int samples = n - start;
        for (int i = start; i < n; ++i) {
            double v = fpsHistory[i]; double norm = v / maxFPS; if (norm < 0) norm = 0; if (norm > 1) norm = 1;
            int x = gx + 6 + (int)((double)(i - start) * (graphW - 12) / double(max(1, samples - 1)));
            int y = gy + graphH - 6 - (int)(norm * (graphH - 12));
            pts.push_back({ x,y });
        }
        HPEN gpen = CreatePen(PS_SOLID, 2, RGB(40, 240, 40)); HPEN oldG = (HPEN)SelectObject(mem, gpen);
        for (size_t i = 1; i < pts.size(); ++i) { MoveToEx(mem, pts[i - 1].x, pts[i - 1].y, NULL); LineTo(mem, pts[i].x, pts[i].y); }
        SelectObject(mem, oldG); DeleteObject(gpen);
    }
    /*
    SetTextColor(mem, RGB(180, 180, 220));
    HFONT small = CreateFontW(12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT oldF = (HFONT)SelectObject(mem, small);
    std::wstring labelHigh = std::wstring(L"") + std::to_wstring((long long)maxFPS);
    TextOutW(mem, gx + 6, gy + 4, labelHigh.c_str(), (int)labelHigh.size());
    TextOutW(mem, gx + 6, gy + graphH - 18, L"0", 1);
    SelectObject(mem, oldF); DeleteObject(small);
    */

    // safer font + text output for the FPS label
    SetTextColor(mem, RGB(180, 180, 220));

    // Build a LOGFONT and create font via CreateFontIndirectW (less error-prone)
    LOGFONTW lf = {};
    lf.lfHeight = -12; // negative = character height in pixels (adjust if you want larger)
    lf.lfWeight = FW_NORMAL;
    lf.lfCharSet = DEFAULT_CHARSET;
    lf.lfQuality = DEFAULT_QUALITY;
    lf.lfPitchAndFamily = DEFAULT_PITCH | FF_SWISS;
    wcsncpy_s(lf.lfFaceName, L"Segoe UI", _TRUNCATE);

    HFONT hfSmall = CreateFontIndirectW(&lf);
    HFONT oldF = (HFONT)SelectObject(mem, hfSmall);

    // Format integer label of maxFPS (rounding)
    int labelFPS = (int)std::lround(maxFPS);
    std::wstring labelHigh = std::to_wstring(labelFPS);

    // Draw the two small labels
    TextOutW(mem, gx + 6, gy + 4, labelHigh.c_str(), (int)labelHigh.length());
    TextOutW(mem, gx + 6, gy + graphH - 18, L"0", 1);

    // restore & cleanup
    SelectObject(mem, oldF);
    if (hfSmall) DeleteObject(hfSmall);

}
static void draw_info_box(HDC mem, int w, int h, bool gpuSuccess, double gpuMs) {
    SetBkMode(mem, TRANSPARENT); SetTextColor(mem, RGB(220, 220, 220));
    HFONT fnt = CreateFontW(16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 10,10,420,220 };
    std::wostringstream ws;
    ws << L"Objects: " << g_objects.size() << L"\n";
    ws << L"GPU mode: " << (g_useGpuIfAvailable ? L"enabled" : L"disabled") << L"\n";
    ws << L"GPU build: " << (gpuSuccess ? L"ok" : L"fallback (CPU)") << L"\n";
    if (gpuSuccess) ws << L"gpu ms: " << to_wstring_utf8(format_double(gpuMs, 3)) << L"\n";
    DrawTextW(mem, ws.str().c_str(), -1, &rc, DT_LEFT | DT_TOP | DT_NOPREFIX);
    SelectObject(mem, old); DeleteObject(fnt);
}

// menu IDs
enum {
    IDM_ADD_TRI = 1001,
    IDM_ADD_CIRC = 1002,
    IDM_ADD_BOX = 1003,
    IDM_RESET = 1004,
    IDM_TOGGLE_GPU = 1005
};

// main entry (Win32)
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
    // register class
    const wchar_t* cls = L"CUDA_GDI_3D_Demo_With_GPU";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.style = CS_OWNDC;
    RegisterClassW(&wc);

    // create window
    HWND hwnd = CreateWindowExW(0, cls, L"CUDA 3D Demo (GPU transforms with CPU fallback)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1000, 720, NULL, NULL, hInst, NULL);
    if (!hwnd) return 1;
    HDC hdc = GetDC(hwnd);

    // create menu
    HMENU hMenu = CreateMenu();
    HMENU hSub = CreatePopupMenu();
    AppendMenuW(hSub, MF_STRING, IDM_ADD_TRI, L"Add Triangle");
    AppendMenuW(hSub, MF_STRING, IDM_ADD_CIRC, L"Add Circle (cylinder)");
    AppendMenuW(hSub, MF_STRING, IDM_ADD_BOX, L"Add Box");
    AppendMenuW(hSub, MF_SEPARATOR, 0, NULL);
    AppendMenuW(hSub, MF_STRING, IDM_RESET, L"Reset Objects");
    AppendMenuW(hSub, MF_STRING, IDM_TOGGLE_GPU, L"Toggle GPU Usage");
    AppendMenuW(hMenu, MF_POPUP, (UINT_PTR)hSub, L"Objects / Add");
    SetMenu(hwnd, hMenu);

    // initial object
    RECT rc; GetClientRect(hwnd, &rc);
    add_random_obj(rc.right - rc.left, rc.bottom - rc.top, OBJ_TRIANGLE);

    // double buffer
    HDC memDC = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, rc.right - rc.left, rc.bottom - rc.top);
    SelectObject(memDC, bmp);

    using clock = std::chrono::steady_clock;
    auto lastFrame = clock::now();
    auto lastSecond = clock::now();
    int frames = 0;
    double displayFPS = 0.0;
    double lastGpuMs = -1.0;
    bool lastGpuSuccess = false;

    std::vector<DrawPoly> polys;

    MSG msg;
    bool running = true;
    while (running) {
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) { running = false; break; }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;

        // timing
        auto now = clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - lastFrame).count();
        if (dt <= 0) dt = 0.016f;
        lastFrame = now;

        // animate rotation speeds
        for (auto& o : g_objects) {
            o.rotX += o.rotSpeedX * dt;
            o.rotY += o.rotSpeedY * dt;
            o.rotZ += o.rotSpeedZ * dt;
        }

        // attempt GPU build if allowed
        bool gpuOk = false; lastGpuMs = -1.0;
        if (g_useGpuIfAvailable) {
            if (gpu_build_polygons_driver_api(g_objects, rc.right - rc.left, rc.bottom - rc.top, g_camYaw, g_camPitch, g_camDistance, polys, &lastGpuMs)) {
                gpuOk = true;
            }
        }

        if (!gpuOk) {
            // CPU fallback
            polys.clear();
            for (const auto& o : g_objects) build_obj_polys_cpu(o, rc.right - rc.left, rc.bottom - rc.top, polys);
        }

        // depth sort (far to near)
        std::sort(polys.begin(), polys.end(), [](const DrawPoly& a, const DrawPoly& b) { return a.depth > b.depth; });

        // draw
        clear_bg(memDC, rc.right - rc.left, rc.bottom - rc.top);
        draw_polys_gdi(memDC, polys);
        draw_info_box(memDC, rc.right - rc.left, rc.bottom - rc.top, gpuOk, lastGpuMs);
        draw_fps_big(memDC, rc.right - rc.left, rc.bottom - rc.top, displayFPS);
        draw_fps_graph(memDC, rc.right - rc.left, rc.bottom - rc.top, g_fpsHistory);
        BitBlt(hdc, 0, 0, rc.right - rc.left, rc.bottom - rc.top, memDC, 0, 0, SRCCOPY);

        // fps counting
        frames++;
        auto nowS = clock::now();
        double dtsec = std::chrono::duration_cast<std::chrono::duration<double>>(nowS - lastSecond).count();
        if (dtsec >= 0.25) {
            displayFPS = double(frames) / dtsec;
            g_fpsHistory.push_back(displayFPS);
            if (g_fpsHistory.size() > 1024) g_fpsHistory.erase(g_fpsHistory.begin(), g_fpsHistory.begin() + (g_fpsHistory.size() - 1024));
            std::ostringstream title; title << "CUDA 3D Demo - Objects=" << g_objects.size() << (g_useGpuIfAvailable ? " GPU=on" : " GPU=off");
            SetWindowTextA(hwnd, title.str().c_str());
            frames = 0;
            lastSecond = nowS;
        }

        // tiny sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
    }

    // cleanup
    DeleteObject(bmp);
    DeleteDC(memDC);
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    return 0;
}

// main wrapper to satisfy linker if needed
int main() {
    HINSTANCE hInst = GetModuleHandleW(NULL);
    return wWinMain(hInst, NULL, GetCommandLineW(), SW_SHOWDEFAULT);
}

#endif


#if 0
// cuda_gdi_demo_autoadd_3d_with_boxes_and_camera.cpp
// Single-file Win32 + GDI demo with CUDA probing (optional).
// - improved host pinned detection (cudaHostAlloc or cudaHostRegister fallback)
// - added Add menu with Triangle / Circle (cylinder) / Box (3D) objects
// - full 3D camera with mouse rotation + wheel zoom
// - polygons depth-sorted and rendered via GDI
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
#include <algorithm>

// ---------------- Dynamic CUDA function typedefs ----------------
typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;
typedef CUresult(__stdcall* cuInit_t)(unsigned int);
typedef CUresult(__stdcall* cuDeviceGetAttribute_t)(int* pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult(__stdcall* cuDeviceGetName_t)(char* name, int len, CUdevice dev);

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
typedef int(__stdcall* cudaRuntimeGetVersion_t)(int*);
typedef int(__stdcall* cudaDriverGetVersion_t)(int*);
typedef int(__stdcall* cudaGetDeviceCount_t)(int*);
typedef int(__stdcall* cudaHostRegister_t)(void*, size_t, unsigned int);
typedef int(__stdcall* cudaHostUnregister_t)(void*);

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
static cuDeviceGetName_t my_cuDeviceGetName = nullptr;

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
static cudaRuntimeGetVersion_t my_cudaRuntimeGetVersion = nullptr;
static cudaDriverGetVersion_t my_cudaDriverGetVersion = nullptr;
static cudaGetDeviceCount_t my_cudaGetDeviceCount = nullptr;
static cudaHostRegister_t my_cudaHostRegister = nullptr;
static cudaHostUnregister_t my_cudaHostUnregister = nullptr;

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

// ---------------- Scene object types ----------------
enum ObjType { OBJ_TRIANGLE = 0, OBJ_CYLINDER = 1, OBJ_BOX = 2 };

// 3D object
struct Obj {
    ObjType type;
    // center position in screen-space coordinates (we'll convert to world with center origin)
    float cx, cy;
    float z;        // depth
    float size;     // base size
    // orientation
    float rotX, rotY, rotZ;
    float rotSpeedX, rotSpeedY, rotSpeedZ;
    // additional parameters for cylinders (segments, thickness)
    int cylSegments;
    float thickness; // for cylinder height or box depth
    COLORREF color;
};

// polygon to draw (screen coords + avg depth)
struct DrawPoly {
    std::vector<POINT> pts;
    float depth; // used for sorting (larger depth -> farther; draw back to front)
    COLORREF color;
};

static std::vector<Obj> g_objects; // scene objects
static std::mt19937 g_rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
static const size_t MAX_OBJECTS = 200000; // safety cap

// Adds random triangle object (3D triangular plate)
static void add_triangle_rand(int winW, int winH) {
    if (g_objects.size() >= MAX_OBJECTS) return;
    std::uniform_real_distribution<float> cx(0.1f * winW, 0.9f * winW);
    std::uniform_real_distribution<float> cy(0.1f * winH, 0.9f * winH);
    std::uniform_real_distribution<float> sz(8.0f, min(winW, winH) * 0.12f);
    std::uniform_real_distribution<float> zpos(winW * 0.2f, winW * 1.2f);
    std::uniform_real_distribution<float> ang(0.0f, 360.0f);
    std::uniform_real_distribution<float> spd(-90.0f, 90.0f);
    std::uniform_real_distribution<float> tz(-8.0f, 8.0f);
    std::uniform_int_distribution<int> col(40, 255);
    Obj o;
    o.type = OBJ_TRIANGLE;
    o.cx = cx(g_rng);
    o.cy = cy(g_rng);
    o.z = zpos(g_rng);
    o.size = sz(g_rng);
    o.rotX = ang(g_rng);
    o.rotY = ang(g_rng);
    o.rotZ = ang(g_rng);
    o.rotSpeedX = spd(g_rng);
    o.rotSpeedY = spd(g_rng);
    o.rotSpeedZ = spd(g_rng);
    o.cylSegments = 0;
    o.thickness = tz(g_rng);
    o.color = RGB(col(g_rng), col(g_rng), col(g_rng));
    g_objects.push_back(o);
}

// Adds random cylinder (used for "circle") - short height
static void add_cylinder_rand(int winW, int winH) {
    if (g_objects.size() >= MAX_OBJECTS) return;
    std::uniform_real_distribution<float> cx(0.1f * winW, 0.9f * winW);
    std::uniform_real_distribution<float> cy(0.1f * winH, 0.9f * winH);
    std::uniform_real_distribution<float> sz(8.0f, min(winW, winH) * 0.12f);
    std::uniform_real_distribution<float> zpos(winW * 0.2f, winW * 1.2f);
    std::uniform_real_distribution<float> ang(0.0f, 360.0f);
    std::uniform_real_distribution<float> spd(-60.0f, 60.0f);
    std::uniform_int_distribution<int> seg(12, 28);
    std::uniform_int_distribution<int> col(40, 255);
    std::uniform_real_distribution<float> height(6.0f, 40.0f);
    Obj o;
    o.type = OBJ_CYLINDER;
    o.cx = cx(g_rng);
    o.cy = cy(g_rng);
    o.z = zpos(g_rng);
    o.size = sz(g_rng);
    o.rotX = ang(g_rng);
    o.rotY = ang(g_rng);
    o.rotZ = ang(g_rng);
    o.rotSpeedX = spd(g_rng);
    o.rotSpeedY = spd(g_rng);
    o.rotSpeedZ = spd(g_rng);
    o.cylSegments = seg(g_rng);
    o.thickness = height(g_rng);
    o.color = RGB(col(g_rng), col(g_rng), col(g_rng));
    g_objects.push_back(o);
}

// Adds random box (true 3D rectangular box)
static void add_box_rand(int winW, int winH) {
    if (g_objects.size() >= MAX_OBJECTS) return;
    std::uniform_real_distribution<float> cx(0.1f * winW, 0.9f * winW);
    std::uniform_real_distribution<float> cy(0.1f * winH, 0.9f * winH);
    std::uniform_real_distribution<float> sz(8.0f, min(winW, winH) * 0.12f);
    std::uniform_real_distribution<float> zpos(winW * 0.2f, winW * 1.2f);
    std::uniform_real_distribution<float> ang(0.0f, 360.0f);
    std::uniform_real_distribution<float> spd(-45.0f, 45.0f);
    std::uniform_real_distribution<float> depth(6.0f, 80.0f);
    std::uniform_int_distribution<int> col(40, 255);
    Obj o;
    o.type = OBJ_BOX;
    o.cx = cx(g_rng);
    o.cy = cy(g_rng);
    o.z = zpos(g_rng);
    o.size = sz(g_rng);
    o.rotX = ang(g_rng);
    o.rotY = ang(g_rng);
    o.rotZ = ang(g_rng);
    o.rotSpeedX = spd(g_rng);
    o.rotSpeedY = spd(g_rng);
    o.rotSpeedZ = spd(g_rng);
    o.cylSegments = 0;
    o.thickness = depth(g_rng);
    o.color = RGB(col(g_rng), col(g_rng), col(g_rng));
    g_objects.push_back(o);
}

static void add_n_objects(int winW, int winH, int n) {
    for (int i = 0; i < n; ++i) {
        if (g_objects.size() >= MAX_OBJECTS) break;
        add_triangle_rand(winW, winH);
    }
}

static void reset_objects() {
    g_objects.clear();
}

// ensure this matches how you call it: clear_bg(HDC mem, int w, int h)
static void clear_bg(HDC mem, int w, int h) {
    if (!mem || w <= 0 || h <= 0) return;
    RECT r = { 0, 0, w, h };
    HBRUSH bg = CreateSolidBrush(RGB(15, 15, 30)); // dark bluish background
    FillRect(mem, &r, bg);
    DeleteObject(bg);
}


// simple local rotations (degrees)
static inline void rotate_xyz(float& x, float& y, float& z, float ax, float ay, float az) {
    float rx = ax * 3.14159265f / 180.0f;
    float ry = ay * 3.14159265f / 180.0f;
    float rz = az * 3.14159265f / 180.0f;
    // X
    float cx = cosf(rx), sx = sinf(rx);
    float y1 = y * cx - z * sx;
    float z1 = y * sx + z * cx;
    y = y1; z = z1;
    // Y
    float cy = cosf(ry), sy = sinf(ry);
    float x1 = x * cy + z * sy;
    float z2 = -x * sy + z * cy;
    x = x1; z = z2;
    // Z
    float cz = cosf(rz), sz = sinf(rz);
    float x2 = x * cz - y * sz;
    float y2 = x * sz + y * cz;
    x = x2; y = y2;
}

// ---------------- Camera state ----------------
static float g_camYaw = 0.0f;      // degrees
static float g_camPitch = 0.0f;    // degrees
static float g_camDistance = 600.0f; // distance scalar; mouse wheel modifies
static float g_camTargetX = 0.0f, g_camTargetY = 0.0f, g_camTargetZ = 0.0f;

// project world point (centered with screen origin 0,0)
static bool project_point(int screenW, int screenH, float wx, float wy, float wz, POINT& out, float& outDepth) {
    // translate world so center of screen is origin
    float x = wx - (float)screenW * 0.5f;
    float y = wy - (float)screenH * 0.5f;
    float z = wz;

    // apply camera rotation (inverse because we rotate world opposite of camera)
    rotate_xyz(x, y, z, -g_camPitch, -g_camYaw, 0.0f);

    // translate by camera distance (camera at z = -g_camDistance)
    float zcam = z + g_camDistance - g_camTargetZ;
    float xcam = x - g_camTargetX;
    float ycam = y - g_camTargetY;

    // avoid behind-camera points
    const float minZ = 1.0f;
    if (zcam < minZ) zcam = minZ;

    // focal length: scale with width and allow zooming via camDistance
    float focal = (float)screenW * 0.8f;

    float sx = (xcam * (focal / zcam)) + (float)screenW * 0.5f;
    float sy = (ycam * (focal / zcam)) + (float)screenH * 0.5f;

    out.x = LONG(sx);
    out.y = LONG(sy);
    outDepth = zcam;
    return true;
}

// build polygons for an Obj into drawPolys (in world coords then projected)
static void build_obj_polys(const Obj& o, int w, int h, std::vector<DrawPoly>& outPolys) {
    // center in world coordinates: keep using screen coordinates as world XY but with center origin in project_point
    // local geometry defined relative to object center
    if (o.type == OBJ_TRIANGLE) {
        // triangular plate with small thickness
        float s = o.size;
        float localX[3] = { 0.0f, -0.6f * s, 0.6f * s };
        float localY[3] = { -1.0f * s, 0.4f * s, 0.4f * s };
        float localZ[3] = { -o.thickness * 0.2f, o.thickness * 0.2f, 0.0f };
        // rotate each vertex by object's rotation and then translate to world
        DrawPoly poly;
        poly.color = o.color;
        poly.pts.resize(3);
        float depths[3];
        for (int i = 0; i < 3; ++i) {
            float x = localX[i], y = localY[i], z = localZ[i];
            rotate_xyz(x, y, z, o.rotX, o.rotY, o.rotZ);
            float wx = o.cx + x;
            float wy = o.cy + y;
            float wz = o.z + z;
            float dep;
            project_point(w, h, wx, wy, wz, poly.pts[i], dep);
            depths[i] = dep;
        }
        // compute average depth
        float avg = (depths[0] + depths[1] + depths[2]) / 3.0f;
        poly.depth = avg;
        outPolys.push_back(std::move(poly));
    }
    else if (o.type == OBJ_BOX) {
        // box centered at (cx,cy,o.z). width and height = o.size, depth = thickness
        float hw = o.size * 0.5f;
        float hh = o.size * 0.5f;
        float hd = o.thickness * 0.5f;
        // 8 vertices of box (local)
        float vx[8] = { -hw, hw, hw, -hw, -hw, hw, hw, -hw };
        float vy[8] = { -hh, -hh, hh, hh, -hh, -hh, hh, hh };
        float vz[8] = { -hd, -hd, -hd, -hd, hd, hd, hd, hd }; // first 4 are back face, last 4 front face
        POINT pv[8];
        float depths[8];
        for (int i = 0; i < 8; ++i) {
            float x = vx[i], y = vy[i], z = vz[i];
            rotate_xyz(x, y, z, o.rotX, o.rotY, o.rotZ);
            float wx = o.cx + x;
            float wy = o.cy + y;
            float wz = o.z + z;
            float d;
            project_point(w, h, wx, wy, wz, pv[i], d);
            depths[i] = d;
        }
        // faces: each face as polygon with color (we could shade based on normal but keep same color)
        const int faceIdx[6][4] = {
            {0,1,2,3}, // back
            {4,5,6,7}, // front
            {0,1,5,4}, // bottom
            {2,3,7,6}, // top
            {1,2,6,5}, // right
            {3,0,4,7}  // left
        };
        for (int f = 0; f < 6; ++f) {
            DrawPoly p;
            p.color = o.color;
            p.pts.resize(4);
            float avg = 0.0f;
            for (int k = 0; k < 4; ++k) {
                int idx = faceIdx[f][k];
                p.pts[k] = pv[idx];
                avg += depths[idx];
            }
            p.depth = avg / 4.0f;
            outPolys.push_back(std::move(p));
        }
    }
    else if (o.type == OBJ_CYLINDER) {
        // cylinder aligned along local Z (extruded along Z), approximated with segments
        int segs = max(6, o.cylSegments);
        float radius = o.size * 0.6f;
        float halfh = o.thickness * 0.5f;
        // compute ring points top and bottom
        std::vector<POINT> topPts(segs), botPts(segs);
        std::vector<float> topDepths(segs), botDepths(segs);
        for (int i = 0; i < segs; ++i) {
            float theta = (float)i * 2.0f * 3.14159265358979f / (float)segs;
            float lx = cosf(theta) * radius;
            float ly = sinf(theta) * radius;
            float lzTop = halfh;
            float lzBot = -halfh;
            float x = lx, y = ly, z = lzTop;
            rotate_xyz(x, y, z, o.rotX, o.rotY, o.rotZ);
            float wx = o.cx + x;
            float wy = o.cy + y;
            float wz = o.z + z;
            float d;
            project_point(w, h, wx, wy, wz, topPts[i], d);
            topDepths[i] = d;

            x = lx; y = ly; z = lzBot;
            rotate_xyz(x, y, z, o.rotX, o.rotY, o.rotZ);
            wx = o.cx + x; wy = o.cy + y; wz = o.z + z;
            project_point(w, h, wx, wy, wz, botPts[i], d);
            botDepths[i] = d;
        }
        // top face (triangle fan)
        {
            DrawPoly p;
            p.color = o.color;
            std::vector<POINT> fan;
            float avg = 0.0f;
            // center
            float x = 0.0f, y = 0.0f, z = halfh;
            rotate_xyz(x, y, z, o.rotX, o.rotY, o.rotZ);
            float wx = o.cx + x, wy = o.cy + y, wz = o.z + z;
            POINT centerPt; float centerDepth;
            project_point(w, h, wx, wy, wz, centerPt, centerDepth);
            avg += centerDepth;
            for (int i = 0; i < segs; ++i) { fan.push_back(topPts[i]); avg += topDepths[i]; }
            // we will triangulate in renderer by drawing as polygon (fan shape)
            p.pts = fan;
            p.depth = avg / (segs + 1);
            outPolys.push_back(std::move(p));
        }
        // bottom face
        {
            DrawPoly p;
            p.color = o.color;
            std::vector<POINT> fan;
            float avg = 0.0f;
            for (int i = 0; i < segs; ++i) { fan.push_back(botPts[i]); avg += botDepths[i]; }
            p.pts = fan;
            p.depth = avg / segs;
            outPolys.push_back(std::move(p));
        }
        // sides: quads between consecutive segments
        for (int i = 0; i < segs; ++i) {
            int ni = (i + 1) % segs;
            DrawPoly p;
            p.color = o.color;
            p.pts.resize(4);
            p.pts[0] = topPts[i];
            p.pts[1] = topPts[ni];
            p.pts[2] = botPts[ni];
            p.pts[3] = botPts[i];
            p.depth = (topDepths[i] + topDepths[ni] + botDepths[ni] + botDepths[i]) * 0.25f;
            outPolys.push_back(std::move(p));
        }
    }
}

// ---------------- Menu command IDs ----------------
enum {
    IDM_ADD_OBJECT = 1001,
    IDM_RESET_OBJECTS = 1002,
    IDM_TOGGLE_AUTOADD = 1003,
    IDM_ADD_100_NOW = 1004,
    // new add menu
    IDM_ADD_TRIANGLE = 2001,
    IDM_ADD_CIRCLE = 2002,
    IDM_ADD_BOX = 2003
};

// ---------------- Globals controlling UI state ----------------
static int g_smCount = -1;                 // from driver API (or -1 if n/a)
static bool g_cudaAvailable = false;
static std::vector<double> g_fpsHistory;   // accumulate history
static const size_t HISTORY_MAX = 1024;
static bool g_autoAdd = false;

// mouse camera interaction
static bool g_mouseDown = false;
static POINT g_mouseStart = { 0,0 };
static float g_startYaw = 0.0f;
static float g_startPitch = 0.0f;

// ---------------- Window Proc ----------------
LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDM_ADD_OBJECT:
        {
            RECT r; GetClientRect(hwnd, &r);
            add_triangle_rand(r.right - r.left, r.bottom - r.top);
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
        case IDM_ADD_TRIANGLE:
        {
            RECT r; GetClientRect(hwnd, &r);
            add_triangle_rand(r.right - r.left, r.bottom - r.top);
        }
        return 0;
        case IDM_ADD_CIRCLE:
        {
            RECT r; GetClientRect(hwnd, &r);
            add_cylinder_rand(r.right - r.left, r.bottom - r.top);
        }
        return 0;
        case IDM_ADD_BOX:
        {
            RECT r; GetClientRect(hwnd, &r);
            add_box_rand(r.right - r.left, r.bottom - r.top);
        }
        return 0;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_LBUTTONDOWN:
    {
        SetCapture(hwnd);
        g_mouseDown = true;
        g_mouseStart.x = LOWORD(lParam);
        g_mouseStart.y = HIWORD(lParam);
        g_startYaw = g_camYaw;
        g_startPitch = g_camPitch;
        return 0;
    }
    case WM_LBUTTONUP:
    {
        ReleaseCapture();
        g_mouseDown = false;
        return 0;
    }
    case WM_MOUSEMOVE:
    {
        if (g_mouseDown) {
            int mx = LOWORD(lParam);
            int my = HIWORD(lParam);
            int dx = mx - g_mouseStart.x;
            int dy = my - g_mouseStart.y;
            g_camYaw = g_startYaw + dx * 0.25f;   // sensitivity
            g_camPitch = g_startPitch + dy * 0.25f;
            if (g_camPitch > 89.0f) g_camPitch = 89.0f;
            if (g_camPitch < -89.0f) g_camPitch = -89.0f;
        }
        return 0;
    }
    case WM_MOUSEWHEEL:
    {
        short delta = GET_WHEEL_DELTA_WPARAM(wParam);
        // zoom multiplicatively
        float factor = (delta > 0) ? 0.9f : 1.1f;
        // scale by number of steps
        int steps = abs(delta) / WHEEL_DELTA;
        for (int i = 0; i < steps; ++i) g_camDistance *= factor;
        if (g_camDistance < 50.0f) g_camDistance = 50.0f;
        if (g_camDistance > 5000.0f) g_camDistance = 5000.0f;
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// ---------------- Info drawing ----------------
static void draw_info_box(HDC mem, int w, int h,
    const std::string& devName, int driverVer, int runtimeVer, int devCount,
    double lastH2D_ms, double lastD2H_ms, double lastD2D_ms,
    size_t freeB, size_t totalB, bool hostPinned)
{
    SetBkMode(mem, TRANSPARENT);
    SetTextColor(mem, RGB(220, 220, 220));
    HFONT fnt = CreateFontW(16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 10, 10, 420, 280 };
    std::wostringstream ws;
    if (!devName.empty()) ws << L"GPU: " << to_wstring_utf8(devName) << L"\n";
    else ws << L"GPU: n/a\n";
    if (g_smCount >= 0) ws << L"SMs: " << g_smCount << L"\n";
    else ws << L"SMs: n/a\n";
    ws << L"Devices: " << devCount << L"\n";
    if (driverVer > 0) ws << L"Driver ver: " << driverVer << L"\n";
    if (runtimeVer > 0) ws << L"Runtime ver: " << runtimeVer << L"\n";
    ws << L"Objects: " << g_objects.size() << L"\n";
    ws << L"AutoAdd: " << (g_autoAdd ? L"ON" : L"OFF") << L"\n";
    ws << L"Host pinned: " << (hostPinned ? L"Yes" : L"No") << L"\n";
    ws << L"H->D ms: " << (lastH2D_ms >= 0 ? to_wstring_utf8(format_double(lastH2D_ms, 3)) : L"n/a") << L"\n";
    ws << L"D->H ms: " << (lastD2H_ms >= 0 ? to_wstring_utf8(format_double(lastD2H_ms, 3)) : L"n/a") << L"\n";
    ws << L"D->D ms: " << (lastD2D_ms >= 0 ? to_wstring_utf8(format_double(lastD2D_ms, 3)) : L"n/a") << L"\n";
    if (totalB > 0) {
        ws << L"GPU mem free: " << (freeB / (1024ULL * 1024ULL)) << L"MB / " << (totalB / (1024ULL * 1024ULL)) << L"MB\n";
    }
    DrawTextW(mem, ws.str().c_str(), -1, &rc, DT_LEFT | DT_TOP | DT_NOPREFIX);
    SelectObject(mem, old);
    DeleteObject(fnt);
}

// FPS big
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

// FPS graph
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

    // compute scale
    double maxFPS = 60.0;
    for (double v : fpsHistory) if (v > maxFPS) maxFPS = v;
    if (maxFPS < 30.0) maxFPS = 30.0;

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

    // labels
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

// ---------------- Main ----------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
    // register class
    const wchar_t* cls = L"CUDA_GDI_MenuGraph_AutoAdd_3D_Advanced";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.style = CS_OWNDC;
    RegisterClassW(&wc);

    // create window
    HWND hwnd = CreateWindowExW(0, cls, L"CUDA GDI Demo (3D, boxes, camera)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1100, 760, NULL, NULL, hInst, NULL);
    if (!hwnd) return 1;
    HDC hdc = GetDC(hwnd);

    // create menus: existing Objects and new Add
    HMENU hMenu = CreateMenu();
    HMENU hSubObjects = CreatePopupMenu();
    AppendMenuW(hSubObjects, MF_STRING, IDM_ADD_OBJECT, L"Add Triangle");
    AppendMenuW(hSubObjects, MF_STRING, IDM_ADD_100_NOW, L"Add 100 Now");
    AppendMenuW(hSubObjects, MF_STRING, IDM_RESET_OBJECTS, L"Reset Objects");
    AppendMenuW(hSubObjects, MF_SEPARATOR, 0, NULL);
    AppendMenuW(hSubObjects, MF_STRING, IDM_TOGGLE_AUTOADD, L"Toggle AutoAdd (100/s)");
    AppendMenuW(hMenu, MF_POPUP, (UINT_PTR)hSubObjects, L"Objects");

    HMENU hSubAdd = CreatePopupMenu();
    AppendMenuW(hSubAdd, MF_STRING, IDM_ADD_TRIANGLE, L"Add Triangle");
    AppendMenuW(hSubAdd, MF_STRING, IDM_ADD_CIRCLE, L"Add Circle (cylinder)");
    AppendMenuW(hSubAdd, MF_STRING, IDM_ADD_BOX, L"Add Box");
    AppendMenuW(hMenu, MF_POPUP, (UINT_PTR)hSubAdd, L"Add");

    SetMenu(hwnd, hMenu);

    // load driver (nvcuda.dll)
    HMODULE hDriver = LoadLibraryW(L"nvcuda.dll");
    if (hDriver) {
        my_cuInit = (cuInit_t)GetProcAddress(hDriver, "cuInit");
        my_cuDeviceGetAttribute = (cuDeviceGetAttribute_t)GetProcAddress(hDriver, "cuDeviceGetAttribute");
        my_cuDeviceGetName = (cuDeviceGetName_t)GetProcAddress(hDriver, "cuDeviceGetName");
        if (!my_cuInit || !my_cuDeviceGetAttribute) { my_cuInit = nullptr; my_cuDeviceGetAttribute = nullptr; }
        else {
            CUresult r = my_cuInit(0);
            if (r == 0) {
                int val = 0;
                // attribute 16 used previously (multiprocessor count)
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
        my_cudaRuntimeGetVersion = (cudaRuntimeGetVersion_t)GetProcAddress(hRuntime, "cudaRuntimeGetVersion");
        my_cudaDriverGetVersion = (cudaDriverGetVersion_t)GetProcAddress(hRuntime, "cudaDriverGetVersion");
        my_cudaGetDeviceCount = (cudaGetDeviceCount_t)GetProcAddress(hRuntime, "cudaGetDeviceCount");
        my_cudaHostRegister = (cudaHostRegister_t)GetProcAddress(hRuntime, "cudaHostRegister");
        my_cudaHostUnregister = (cudaHostUnregister_t)GetProcAddress(hRuntime, "cudaHostUnregister");
    }

    // prepare 4MB host buffer (try pinned). Improved: try cudaHostAlloc, else malloc + cudaHostRegister
    const size_t TRANSFER_BYTES = 4 * 1024 * 1024;
    void* hostBuf = nullptr; bool hostPinned = false;
    bool hostRegistered = false;
    if (my_cudaHostAlloc) {
        if (my_cudaHostAlloc(&hostBuf, TRANSFER_BYTES, 0) == 0 && hostBuf) {
            hostPinned = true;
        }
        else {
            hostBuf = nullptr;
        }
    }
    if (!hostBuf) {
        // fallback to malloc + cudaHostRegister (if available)
        hostBuf = malloc(TRANSFER_BYTES);
        if (hostBuf && my_cudaHostRegister) {
            // flags: 0 = default
            if (my_cudaHostRegister(hostBuf, TRANSFER_BYTES, 0) == 0) {
                hostPinned = true;
                hostRegistered = true;
            }
        }
    }
    if (hostBuf) memset(hostBuf, 0xA5, TRANSFER_BYTES);

    // device buffers: devBuf and devBuf2 (for D2D test)
    void* devBuf = nullptr; bool haveDevBuf = false;
    void* devBuf2 = nullptr; bool haveDevBuf2 = false;
    if (my_cudaMalloc) {
        if (my_cudaMalloc(&devBuf, TRANSFER_BYTES) == 0 && devBuf) haveDevBuf = true;
        if (my_cudaMalloc(&devBuf2, TRANSFER_BYTES) == 0 && devBuf2) haveDevBuf2 = true;
    }

    // events
    void* evStart = nullptr; void* evStop = nullptr; bool haveEvents = false;
    if (my_cudaEventCreate && my_cudaEventRecord && my_cudaEventElapsedTime && my_cudaEventSynchronize && my_cudaEventDestroy) {
        if (my_cudaEventCreate(&evStart) == 0 && my_cudaEventCreate(&evStop) == 0) haveEvents = true;
        else { if (evStart) my_cudaEventDestroy(evStart); if (evStop) my_cudaEventDestroy(evStop); evStart = evStop = nullptr; }
    }

    // query additional CUDA info we added
    std::string deviceName;
    int driverVer = 0, runtimeVer = 0;
    int deviceCount = 0;
    if (my_cudaDriverGetVersion) my_cudaDriverGetVersion(&driverVer);
    if (my_cudaRuntimeGetVersion) my_cudaRuntimeGetVersion(&runtimeVer);
    if (my_cudaGetDeviceCount) my_cudaGetDeviceCount(&deviceCount);
    if (my_cuDeviceGetName) {
        char tmpName[256] = { 0 };
        if (my_cuDeviceGetName(tmpName, sizeof(tmpName), 0) == 0) {
            deviceName = std::string(tmpName);
        }
    }

    // initial objects (one triangle)
    RECT rc; GetClientRect(hwnd, &rc);
    add_triangle_rand(rc.right - rc.left, rc.bottom - rc.top);

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
    double lastH2D_ms = -1.0, lastD2H_ms = -1.0, lastD2D_ms = -1.0;
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

        // update objects (rotations)
        auto nowFrame = clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(nowFrame - lastFrame).count();
        if (dt <= 0) dt = 0.016f;
        lastFrame = nowFrame;
        for (auto& o : g_objects) {
            o.rotX += o.rotSpeedX * dt;
            o.rotY += o.rotSpeedY * dt;
            o.rotZ += o.rotSpeedZ * dt;
        }

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

        // build polygons for all objects
        std::vector<DrawPoly> allPolys;
        allPolys.reserve(g_objects.size() * 4);
        for (const auto& o : g_objects) {
            build_obj_polys(o, gbuf.w, gbuf.h, allPolys);
        }
        // depth sort back-to-front (farther first)
        std::sort(allPolys.begin(), allPolys.end(), [](const DrawPoly& a, const DrawPoly& b) {
            return a.depth > b.depth; // larger depth = farther; draw farther first
            });

        // draw background
        HDC mem = gbuf.memDC;
        clear_bg(mem, gbuf.w, gbuf.h);

        // draw polygons in order
        for (const auto& p : allPolys) {
            if (p.pts.empty()) continue;
            HBRUSH brush = CreateSolidBrush(p.color);
            HBRUSH oldB = (HBRUSH)SelectObject(mem, brush);
            HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
            HPEN oldP = (HPEN)SelectObject(mem, pen);
            Polygon(mem, p.pts.data(), (int)p.pts.size());
            SelectObject(mem, oldB); DeleteObject(brush);
            SelectObject(mem, oldP); DeleteObject(pen);
        }

        // mem info
        if (my_cudaMemGetInfo) {
            size_t freeB = 0, totalB = 0;
            if (my_cudaMemGetInfo(&freeB, &totalB) == 0) { lastFree = freeB; lastTotal = totalB; }
        }

        // draw overlays
        draw_info_box(mem, gbuf.w, gbuf.h, deviceName, driverVer, runtimeVer, deviceCount,
            lastH2D_ms, lastD2H_ms, lastD2D_ms, lastFree, lastTotal, hostPinned);

        draw_fps_big(mem, gbuf.w, gbuf.h, displayFPS);
        draw_fps_graph(mem, gbuf.w, gbuf.h, g_fpsHistory);

        // blit
        BitBlt(hdc, 0, 0, gbuf.w, gbuf.h, gbuf.memDC, 0, 0, SRCCOPY);

        // CUDA transfer measurements (if available)
        if (haveDevBuf && my_cudaMemcpy) {
            if (haveEvents) {
                // Host -> Device
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                float ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastH2D_ms = ms;

                // Device -> Host
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastD2H_ms = ms;

                // Device -> Device
                if (haveDevBuf2) {
                    my_cudaEventRecord(evStart, NULL);
                    my_cudaMemcpy(devBuf2, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToDevice);
                    my_cudaEventRecord(evStop, NULL);
                    my_cudaEventSynchronize(evStop);
                    ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastD2D_ms = ms;
                }
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

                if (haveDevBuf2) {
                    t0 = std::chrono::high_resolution_clock::now();
                    my_cudaMemcpy(devBuf2, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToDevice);
                    t1 = std::chrono::high_resolution_clock::now();
                    lastD2D_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
                }
            }
        }

        // frames and per-second updates
        frames++;
        auto now = clock::now();
        double dtsec = std::chrono::duration_cast<std::chrono::duration<double>>(now - lastSecond).count();
        if (dtsec >= 0.25) {
            displayFPS = double(frames) / dtsec;
            g_fpsHistory.push_back(displayFPS);
            if (g_fpsHistory.size() > HISTORY_MAX) g_fpsHistory.erase(g_fpsHistory.begin(), g_fpsHistory.begin() + (g_fpsHistory.size() - HISTORY_MAX));
            std::ostringstream title;
            title << "CUDA GDI Demo 3D - Objects=" << g_objects.size() << " | SMs=" << (g_smCount >= 0 ? std::to_string(g_smCount) : "n/a") << (g_autoAdd ? " | AutoAdd=ON" : "");
            SetWindowTextW(hwnd, to_wstring_utf8(title.str()).c_str());

            static double accumLog = 0.0;
            accumLog += dtsec;
            if (accumLog >= 1.0) {
                std::ostringstream log;
                log << "FPS=" << std::fixed << std::setprecision(1) << displayFPS
                    << " Objects=" << g_objects.size()
                    << " SMs=" << (g_smCount >= 0 ? std::to_string(g_smCount) : "n/a")
                    << " H2Dms=" << (lastH2D_ms >= 0 ? format_double(lastH2D_ms, 3) : "n/a")
                    << " D2Hms=" << (lastD2H_ms >= 0 ? format_double(lastD2H_ms, 3) : "n/a")
                    << " D2Dms=" << (lastD2D_ms >= 0 ? format_double(lastD2D_ms, 3) : "n/a")
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
    if (devBuf2 && my_cudaFree) my_cudaFree(devBuf2);
    if (devBuf && my_cudaFree) my_cudaFree(devBuf);
    if (hostBuf) {
        if (hostPinned) {
            if (hostRegistered && my_cudaHostUnregister) {
                my_cudaHostUnregister(hostBuf);
            }
            else if (my_cudaFreeHost && !hostRegistered) {
                my_cudaFreeHost(hostBuf);
            }
            else {
                // if pinned but no unregister/free available, fall back to free
                free(hostBuf);
            }
        }
        else {
            free(hostBuf);
        }
    }
    if (hRuntime) FreeLibrary(hRuntime);
    if (hDriver) FreeLibrary(hDriver);
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    return 0;
}
#endif


#if 0 
// cuda_gdi_demo_autoadd_3d.cpp
// Single-file Win32 + GDI demo with CUDA probing (optional).
// - added more CUDA probing (device name, driver/runtime versions, device count)
// - measures Host->Device, Device->Host, Device->Device transfer times (ms)
// - shows GPU memory free/total
// - triangles are now 3D with a simple perspective projection
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
typedef CUresult(__stdcall* cuDeviceGetName_t)(char* name, int len, CUdevice dev);

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
typedef int(__stdcall* cudaRuntimeGetVersion_t)(int*);
typedef int(__stdcall* cudaDriverGetVersion_t)(int*);
typedef int(__stdcall* cudaGetDeviceCount_t)(int*);

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
static cuDeviceGetName_t my_cuDeviceGetName = nullptr;

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
static cudaRuntimeGetVersion_t my_cudaRuntimeGetVersion = nullptr;
static cudaDriverGetVersion_t my_cudaDriverGetVersion = nullptr;
static cudaGetDeviceCount_t my_cudaGetDeviceCount = nullptr;

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

// ---------------- Scene objects (3D) ----------------
struct Obj {
    // 3D position & orientation
    float cx, cy;   // screen center reference (2D base position)
    float z;        // depth (>0)
    float size;     // base size scale
    float rotY;     // rotation around Y (degrees)
    float rotX;     // rotation around X
    float rotZ;     // rotation around Z
    float rotSpeedX, rotSpeedY, rotSpeedZ;
    float zSpeed;   // depth change speed (for slight parallax)
    COLORREF color;
};

static std::vector<Obj> g_objects; // scene objects
static std::mt19937 g_rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
static const size_t MAX_OBJECTS = 200000; // safety cap

// Adds an object with random parameters centered randomly in the window (3D)
static void add_object(int winW, int winH) {
    if (g_objects.size() >= MAX_OBJECTS) return;
    std::uniform_real_distribution<float> ang(0.0f, 360.0f);
    std::uniform_real_distribution<float> spd(-90.0f, 90.0f); // deg/s
    std::uniform_real_distribution<float> cx(0.1f * winW, 0.9f * winW);
    std::uniform_real_distribution<float> cy(0.1f * winH, 0.9f * winH);
    std::uniform_real_distribution<float> sz(10.0f, min(winW, winH) * 0.12f);
    std::uniform_real_distribution<float> zpos(winW * 0.2f, winW * 1.2f); // depth relative to width
    std::uniform_real_distribution<float> zspd(-5.0f, 5.0f);
    std::uniform_int_distribution<int> col(40, 255);
    Obj o;
    o.cx = cx(g_rng);
    o.cy = cy(g_rng);
    o.size = sz(g_rng);
    o.z = zpos(g_rng);
    o.rotX = ang(g_rng);
    o.rotY = ang(g_rng);
    o.rotZ = ang(g_rng);
    o.rotSpeedX = spd(g_rng);
    o.rotSpeedY = spd(g_rng);
    o.rotSpeedZ = spd(g_rng);
    o.zSpeed = zspd(g_rng);
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

// simple rotation helpers (degrees)
static inline void rotate_point_xyz(float& x, float& y, float& z, float ax, float ay, float az) {
    // convert degrees to radians
    float rx = ax * 3.14159265f / 180.0f;
    float ry = ay * 3.14159265f / 180.0f;
    float rz = az * 3.14159265f / 180.0f;
    // rotate X
    float cy = cosf(rx), sy = sinf(rx);
    float y1 = y * cy - z * sy;
    float z1 = y * sy + z * cy;
    y = y1; z = z1;
    // rotate Y
    float cx = cosf(ry), sx = sinf(ry);
    float x1 = x * cx + z * sx;
    float z2 = -x * sx + z * cx;
    x = x1; z = z2;
    // rotate Z
    float cz = cosf(rz), sz = sinf(rz);
    float x2 = x * cz - y * sz;
    float y2 = x * sz + y * cz;
    x = x2; y = y2;
}

// draw filled triangle projected from 3D
static void draw_triangle_instance_3d(HDC mem, int w, int h, const Obj& o) {
    // Local triangle in object local space (3 vertices)
    // We'll use a triangular plate with slight depth variation to appear 3D.
    float s = o.size;
    float vx[3] = { 0.0f, -0.6f * s, 0.6f * s };
    float vy[3] = { -1.0f * s, 0.4f * s, 0.4f * s };
    float vz[3] = { -0.1f * s, 0.1f * s, 0.0f }; // small differences in z
    POINT pts[3];

    // simple perspective projection parameters
    // focal length: bigger => less perspective. scale with width for stability.
    float focal = (float)w * 0.75f;

    for (int i = 0; i < 3; ++i) {
        float x = vx[i];
        float y = vy[i];
        float z = vz[i];

        // apply rotation
        float xr = x, yr = y, zr = z;
        rotate_point_xyz(xr, yr, zr, o.rotX, o.rotY, o.rotZ);

        // translate to world position (object center), add object's z
        float worldX = xr + o.cx;
        float worldY = yr + o.cy;
        float worldZ = zr + o.z;

        // perspective projection (avoid div by zero)
        float depth = worldZ + 1e-3f;
        float px = ((worldX - (float)w * 0.5f) * (focal / depth)) + (float)w * 0.5f;
        float py = ((worldY - (float)h * 0.5f) * (focal / depth)) + (float)h * 0.5f;

        pts[i].x = LONG(px);
        pts[i].y = LONG(py);
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

// overlay: show SM count, object count, autoadd status + CUDA info
static void draw_info_box(HDC mem, int w, int h, int smCount, size_t objCount, bool autoAdd,
    const std::string& devName, int driverVer, int runtimeVer, int devCount,
    double lastH2D_ms, double lastD2H_ms, double lastD2D_ms,
    size_t freeB, size_t totalB, bool hostPinned)
{
    SetBkMode(mem, TRANSPARENT);
    SetTextColor(mem, RGB(220, 220, 220));
    HFONT fnt = CreateFontW(16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HFONT old = (HFONT)SelectObject(mem, fnt);
    RECT rc = { 10, 10, 420, 260 };
    std::wostringstream ws;
    if (!devName.empty()) ws << L"GPU: " << to_wstring_utf8(devName) << L"\n";
    else ws << L"GPU: n/a\n";
    if (smCount >= 0) ws << L"SMs: " << smCount << L"\n";
    else ws << L"SMs: n/a\n";
    ws << L"Devices: " << devCount << L"\n";
    if (driverVer > 0) ws << L"Driver ver: " << driverVer << L"\n";
    if (runtimeVer > 0) ws << L"Runtime ver: " << runtimeVer << L"\n";
    ws << L"Objects: " << objCount << L"\n";
    ws << L"AutoAdd: " << (autoAdd ? L"ON" : L"OFF") << L"\n";
    ws << L"Host pinned: " << (hostPinned ? L"Yes" : L"No") << L"\n";
    ws << L"H->D ms: " << (lastH2D_ms >= 0 ? to_wstring_utf8(format_double(lastH2D_ms, 3)) : L"n/a") << L"\n";
    ws << L"D->H ms: " << (lastD2H_ms >= 0 ? to_wstring_utf8(format_double(lastD2H_ms, 3)) : L"n/a") << L"\n";
    ws << L"D->D ms: " << (lastD2D_ms >= 0 ? to_wstring_utf8(format_double(lastD2D_ms, 3)) : L"n/a") << L"\n";
    if (totalB > 0) {
        ws << L"GPU mem free: " << (freeB / (1024ULL * 1024ULL)) << L"MB / " << (totalB / (1024ULL * 1024ULL)) << L"MB\n";
    }
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
    const wchar_t* cls = L"CUDA_GDI_MenuGraph_AutoAdd_3D";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.style = CS_OWNDC;
    RegisterClassW(&wc);

    // create window
    HWND hwnd = CreateWindowExW(0, cls, L"CUDA GDI Demo (3D triangles, transfer timing)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
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
        my_cuDeviceGetName = (cuDeviceGetName_t)GetProcAddress(hDriver, "cuDeviceGetName");
        if (!my_cuInit || !my_cuDeviceGetAttribute) { my_cuInit = nullptr; my_cuDeviceGetAttribute = nullptr; }
        else {
            CUresult r = my_cuInit(0);
            if (r == 0) {
                int val = 0;
                // attribute 16 used previously (multiprocessor count) - keep same check
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
        my_cudaRuntimeGetVersion = (cudaRuntimeGetVersion_t)GetProcAddress(hRuntime, "cudaRuntimeGetVersion");
        my_cudaDriverGetVersion = (cudaDriverGetVersion_t)GetProcAddress(hRuntime, "cudaDriverGetVersion");
        my_cudaGetDeviceCount = (cudaGetDeviceCount_t)GetProcAddress(hRuntime, "cudaGetDeviceCount");
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

    // device buffers: devBuf and devBuf2 (for D2D test)
    void* devBuf = nullptr; bool haveDevBuf = false;
    void* devBuf2 = nullptr; bool haveDevBuf2 = false;
    if (my_cudaMalloc) {
        if (my_cudaMalloc(&devBuf, TRANSFER_BYTES) == 0 && devBuf) haveDevBuf = true;
        if (my_cudaMalloc(&devBuf2, TRANSFER_BYTES) == 0 && devBuf2) haveDevBuf2 = true;
    }

    // events
    void* evStart = nullptr; void* evStop = nullptr; bool haveEvents = false;
    if (my_cudaEventCreate && my_cudaEventRecord && my_cudaEventElapsedTime && my_cudaEventSynchronize && my_cudaEventDestroy) {
        if (my_cudaEventCreate(&evStart) == 0 && my_cudaEventCreate(&evStop) == 0) haveEvents = true;
        else { if (evStart) my_cudaEventDestroy(evStart); if (evStop) my_cudaEventDestroy(evStop); evStart = evStop = nullptr; }
    }

    // query additional CUDA info we added
    std::string deviceName;
    int driverVer = 0, runtimeVer = 0;
    int deviceCount = 0;
    if (my_cudaDriverGetVersion) my_cudaDriverGetVersion(&driverVer);
    if (my_cudaRuntimeGetVersion) my_cudaRuntimeGetVersion(&runtimeVer);
    if (my_cudaGetDeviceCount) my_cudaGetDeviceCount(&deviceCount);
    if (my_cuDeviceGetName) {
        char tmpName[256] = { 0 };
        if (my_cuDeviceGetName(tmpName, sizeof(tmpName), 0) == 0) {
            deviceName = std::string(tmpName);
        }
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
    double lastH2D_ms = -1.0, lastD2H_ms = -1.0, lastD2D_ms = -1.0;
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

        // update objects (rotations + depth)
        auto nowFrame = clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(nowFrame - lastFrame).count();
        if (dt <= 0) dt = 0.016f;
        lastFrame = nowFrame;
        for (auto& o : g_objects) {
            o.rotX += o.rotSpeedX * dt;
            o.rotY += o.rotSpeedY * dt;
            o.rotZ += o.rotSpeedZ * dt;
            o.z += o.zSpeed * dt;
            // clamp depth to reasonable range
            if (o.z < 50.0f) o.z = 50.0f;
            if (o.z > (float)gbuf.w * 2.5f) o.z = (float)gbuf.w * 2.5f;
        }

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
        for (const auto& o : g_objects) draw_triangle_instance_3d(gbuf.memDC, gbuf.w, gbuf.h, o);

        // mem info (fill lastFree/lastTotal)
        if (my_cudaMemGetInfo) {
            size_t freeB = 0, totalB = 0;
            if (my_cudaMemGetInfo(&freeB, &totalB) == 0) { lastFree = freeB; lastTotal = totalB; }
        }

        // draw info box (SMs + objects + autoadd status + CUDA info)
        draw_info_box(gbuf.memDC, gbuf.w, gbuf.h, g_smCount, g_objects.size(), g_autoAdd,
            deviceName, driverVer, runtimeVer, deviceCount,
            lastH2D_ms, lastD2H_ms, lastD2D_ms,
            lastFree, lastTotal, hostPinned);

        // draw FPS big on animation screen (center)
        draw_fps_big(gbuf.memDC, gbuf.w, gbuf.h, displayFPS);

        // draw FPS graph in lower-right
        draw_fps_graph(gbuf.memDC, gbuf.w, gbuf.h, g_fpsHistory);

        // blit
        BitBlt(hdc, 0, 0, gbuf.w, gbuf.h, gbuf.memDC, 0, 0, SRCCOPY);

        // CUDA transfer measurements (if available)
        if (haveDevBuf && my_cudaMemcpy) {
            if (haveEvents) {
                // Host -> Device
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                float ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastH2D_ms = ms;

                // Device -> Host
                my_cudaEventRecord(evStart, NULL);
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                my_cudaEventRecord(evStop, NULL);
                my_cudaEventSynchronize(evStop);
                ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastD2H_ms = ms;

                // Device -> Device (if second buffer exists)
                if (haveDevBuf2) {
                    my_cudaEventRecord(evStart, NULL);
                    my_cudaMemcpy(devBuf2, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToDevice);
                    my_cudaEventRecord(evStop, NULL);
                    my_cudaEventSynchronize(evStop);
                    ms = 0.0f; if (my_cudaEventElapsedTime(&ms, evStart, evStop) == 0) lastD2D_ms = ms;
                }
            }
            else {
                // fallback to host timers (coarse)
                auto t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(devBuf, hostBuf, TRANSFER_BYTES, cudaMemcpyHostToDevice);
                auto t1 = std::chrono::high_resolution_clock::now();
                lastH2D_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;

                t0 = std::chrono::high_resolution_clock::now();
                my_cudaMemcpy(hostBuf, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToHost);
                t1 = std::chrono::high_resolution_clock::now();
                lastD2H_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;

                if (haveDevBuf2) {
                    t0 = std::chrono::high_resolution_clock::now();
                    my_cudaMemcpy(devBuf2, devBuf, TRANSFER_BYTES, cudaMemcpyDeviceToDevice);
                    t1 = std::chrono::high_resolution_clock::now();
                    lastD2D_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 0.001;
                }
            }
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
                    << " H2Dms=" << (lastH2D_ms >= 0 ? format_double(lastH2D_ms, 3) : "n/a")
                    << " D2Hms=" << (lastD2H_ms >= 0 ? format_double(lastD2H_ms, 3) : "n/a")
                    << " D2Dms=" << (lastD2D_ms >= 0 ? format_double(lastD2D_ms, 3) : "n/a")
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
    if (devBuf2 && my_cudaFree) my_cudaFree(devBuf2);
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
