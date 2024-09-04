#include <windows.h>
#include <mfapi.h>
#include <mfobjects.h>
#include <mfidl.h>
#include <mfplay.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>
#include <algorithm>

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfplay.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

#define VIDEO_WIDTH 640
#define VIDEO_HEIGHT 480
#define VIDEO_FPS 30
#define VIDEO_DURATION_SECONDS 10

// DirectX 11 global variables
ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pImmediateContext = nullptr;

void InitializeDirectX11() {
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &g_pd3dDevice,
        &featureLevel,
        &g_pImmediateContext
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create DirectX 11 device.");
    }
}

void CleanupDirectX11() {
    if (g_pImmediateContext) g_pImmediateContext->Release();
    if (g_pd3dDevice) g_pd3dDevice->Release();
}

void InitializeMediaFoundation() {
    HRESULT hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to initialize Media Foundation.");
    }
}

void ShutdownMediaFoundation() {
    MFShutdown();
}

IMFSinkWriter* CreateSinkWriter(const std::wstring& outputFilename, IMFMediaType** pVideoTypeOut) {
    IMFSinkWriter* pSinkWriter = NULL;
    IMFMediaType* pVideoTypeIn = NULL;

    HRESULT hr = MFCreateSinkWriterFromURL(outputFilename.c_str(), NULL, NULL, &pSinkWriter);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create Sink Writer.");
    }

    hr = MFCreateMediaType(&pVideoTypeIn);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create input media type.");
    }

    hr = pVideoTypeIn->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    hr = pVideoTypeIn->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    hr = pVideoTypeIn->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    hr = MFSetAttributeSize(pVideoTypeIn, MF_MT_FRAME_SIZE, VIDEO_WIDTH, VIDEO_HEIGHT);
    hr = MFSetAttributeRatio(pVideoTypeIn, MF_MT_FRAME_RATE, VIDEO_FPS, 1);
    hr = MFSetAttributeRatio(pVideoTypeIn, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to set input media type attributes.");
    }

    hr = pSinkWriter->SetInputMediaType(0, pVideoTypeIn, NULL);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to set input media type.");
    }

    hr = MFCreateMediaType(pVideoTypeOut);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create output media type.");
    }

    hr = (*pVideoTypeOut)->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    hr = (*pVideoTypeOut)->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
    hr = (*pVideoTypeOut)->SetUINT32(MF_MT_AVG_BITRATE, 800000);
    hr = MFSetAttributeSize(*pVideoTypeOut, MF_MT_FRAME_SIZE, VIDEO_WIDTH, VIDEO_HEIGHT);
    hr = MFSetAttributeRatio(*pVideoTypeOut, MF_MT_FRAME_RATE, VIDEO_FPS, 1);
    hr = MFSetAttributeRatio(*pVideoTypeOut, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    hr = (*pVideoTypeOut)->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to set output media type attributes.");
    }

    hr = pSinkWriter->AddStream(*pVideoTypeOut, NULL);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to add video stream.");
    }

    hr = pSinkWriter->BeginWriting();
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to begin writing.");
    }

    pVideoTypeIn->Release();

    return pSinkWriter;
}

void WriteFrame(IMFSinkWriter* pSinkWriter, DWORD streamIndex, const BYTE* pData, LONGLONG rtStart) {
    IMFSample* pSample = NULL;
    IMFMediaBuffer* pBuffer = NULL;

    const LONG cbWidth = 4 * VIDEO_WIDTH;
    const DWORD cbBuffer = cbWidth * VIDEO_HEIGHT;

    HRESULT hr = MFCreateMemoryBuffer(cbBuffer, &pBuffer);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create memory buffer.");
    }

    BYTE* pBufferData = NULL;
    hr = pBuffer->Lock(&pBufferData, NULL, NULL);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to lock buffer.");
    }

    memcpy(pBufferData, pData, cbBuffer);

    hr = pBuffer->Unlock();
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to unlock buffer.");
    }

    hr = pBuffer->SetCurrentLength(cbBuffer);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to set buffer length.");
    }

    hr = MFCreateSample(&pSample);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create sample.");
    }

    hr = pSample->AddBuffer(pBuffer);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to add buffer to sample.");
    }

    hr = pSample->SetSampleTime(rtStart);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to set sample time.");
    }

    hr = pSample->SetSampleDuration(10 * 1000 * 1000 / VIDEO_FPS);  // 10 million ticks per second
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to set sample duration.");
    }

    hr = pSinkWriter->WriteSample(streamIndex, pSample);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to write sample.");
    }

    pSample->Release();
    pBuffer->Release();
}

void GenerateVideo() {
    InitializeDirectX11();
    InitializeMediaFoundation();

    IMFSinkWriter* pSinkWriter = NULL;
    IMFMediaType* pVideoTypeOut = NULL;

    try {
        pSinkWriter = CreateSinkWriter(L"C:\\Windows\\Temp\\random_video.mp4", &pVideoTypeOut);

        LONGLONG rtStart = 0;
        const LONG cbWidth = 4 * VIDEO_WIDTH;
        const DWORD cbBuffer = cbWidth * VIDEO_HEIGHT;
        std::vector<BYTE> frameData(cbBuffer);

        for (int i = 0; i < VIDEO_DURATION_SECONDS * VIDEO_FPS; ++i) {
            // Generate random frame data
            std::generate(frameData.begin(), frameData.end(), []() { return static_cast<BYTE>(rand() % 256); });
            WriteFrame(pSinkWriter, 0, frameData.data(), rtStart);
            rtStart += 10 * 1000 * 1000 / VIDEO_FPS;
        }

        // Finalize the video encoding
        HRESULT hr = pSinkWriter->Finalize();
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to finalize the Sink Writer.");
        }

    }
    catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    if (pVideoTypeOut) pVideoTypeOut->Release();
    if (pSinkWriter) pSinkWriter->Release();

    ShutdownMediaFoundation();
    CleanupDirectX11();
}

int main() {
    srand(static_cast<unsigned int>(time(NULL)));
    GenerateVideo();

    std::cout << "Random video generated as random_video.mp4" << std::endl;

    return 0;
}
