#include <Windows.h>
#include <mmdeviceapi.h>
#include <endpointvolume.h>
#include <audioclient.h>
#include <iostream>
#include <comdef.h>

#include <Functiondiscoverykeys_devpkey.h>

#include <avrt.h>

#include <fstream>
#include <stdint.h>


#pragma comment(lib, "avrt.lib")

const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);
const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);

#define REFTIMES_PER_SEC  10000000

// WAV file header structure
struct WAVHeader {
    char riff[4] = { 'R', 'I', 'F', 'F' };      // "RIFF"
    uint32_t fileSize;                        // File size in bytes
    char wave[4] = { 'W', 'A', 'V', 'E' };      // "WAVE"
    char fmt[4] = { 'f', 'm', 't', ' ' };       // "fmt "
    uint32_t fmtSize = 16;                    // Size of the fmt chunk
    uint16_t audioFormat = 1;                 // PCM format
    uint16_t numChannels;                     // Number of channels
    uint32_t sampleRate;                      // Sample rate
    uint32_t byteRate;                        // Byte rate (sampleRate * numChannels * bitsPerSample / 8)
    uint16_t blockAlign;                      // Block align (numChannels * bitsPerSample / 8)
    uint16_t bitsPerSample;                   // Bits per sample
    char data[4] = { 'd', 'a', 't', 'a' };      // "data"
    uint32_t dataSize;                        // Data size in bytes
};

void WriteWAVHeader(std::ofstream& file, WAVEFORMATEX* pwfx, uint32_t dataSize) {
    WAVHeader header;
    header.numChannels = pwfx->nChannels;
    header.sampleRate = pwfx->nSamplesPerSec;
    header.bitsPerSample = pwfx->wBitsPerSample;
    header.byteRate = pwfx->nAvgBytesPerSec;
    header.blockAlign = pwfx->nBlockAlign;
    header.dataSize = dataSize;
    header.fileSize = sizeof(WAVHeader) - 8 + dataSize;  // -8 to exclude "RIFF" and fileSize fields

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
}


void RecordAudio() {
    CoInitialize(NULL);

    // Get default audio capture device
    IMMDeviceEnumerator* pEnumerator = NULL;
    IMMDevice* pDevice = NULL;
    IAudioClient* pAudioClient = NULL;
    IAudioCaptureClient* pCaptureClient = NULL;

    CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator,
        (void**)&pEnumerator);
    pEnumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &pDevice);

    std::cout << " test 1 " << pDevice << std::endl;

    // Initialize audio client
    pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pAudioClient);
    WAVEFORMATEX* pwfx = NULL;
    pAudioClient->GetMixFormat(&pwfx);
    //pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, 0, 0, pwfx, NULL);

    REFERENCE_TIME hnsRequestedDuration = REFTIMES_PER_SEC;
    pAudioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        0,
        hnsRequestedDuration,
        0,
        pwfx,
        NULL);

    pAudioClient->Start();

    std::cout << " test 2 " << pAudioClient << std::endl;

   

    // Get the size of the allocated buffer.
    UINT32 bufferFrameCount;
    HRESULT hr = pAudioClient->GetBufferSize(&bufferFrameCount);

    std::cout << "test 2 buf " << bufferFrameCount << std::endl;

    // Get capture client
    //pAudioClient->GetService(IID_PPV_ARGS(&pCaptureClient));
    pAudioClient->GetService(
        IID_IAudioCaptureClient,
        (void**)&pCaptureClient);

    std::cout << " test 3 " << pCaptureClient << std::endl;


    std::ofstream outFile("C:\\Windows\\Temp\\out.wav", std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file.\n";
        return;
    }

    // Placeholder for the header, we will write the actual header later
    WAVHeader header = {};
    outFile.write(reinterpret_cast<const char*>(&header), sizeof(header));

    UINT32 packetLength = 0;
    uint32_t totalDataSize = 0;  // To track the total data size for the header

    // Start recording loop
    while (true) {
        pCaptureClient->GetNextPacketSize(&packetLength);

        while (packetLength != 0) {
            BYTE* pData;
            UINT32 numFramesAvailable;
            DWORD flags;

            // Get audio data
            HRESULT hr = pCaptureClient->GetBuffer(&pData, &numFramesAvailable, &flags, NULL, NULL);
            if (FAILED(hr)) {
                std::cerr << "Failed to get buffer.\n";
                return;
            }

            UINT32 bytesToWrite = numFramesAvailable * pwfx->nBlockAlign;

            // Write audio data to file
            outFile.write(reinterpret_cast<const char*>(pData), bytesToWrite);
            totalDataSize += bytesToWrite;

            // Release the buffer
            pCaptureClient->ReleaseBuffer(numFramesAvailable);
            pCaptureClient->GetNextPacketSize(&packetLength);
        }

        // Stop after a certain duration (e.g., 10 seconds) or based on another condition
        if (totalDataSize > pwfx->nAvgBytesPerSec * 10) {  // Example: Record 10 seconds
            break;
        }
    }

    // Update the header with the correct data size
    outFile.seekp(0, std::ios::beg);
    WriteWAVHeader(outFile, pwfx, totalDataSize);

    outFile.close();

    // Cleanup
    pAudioClient->Stop();
    CoTaskMemFree(pwfx);
    pCaptureClient->Release();
    pAudioClient->Release();
    pDevice->Release();
    pEnumerator->Release();
    CoUninitialize();
}


void ListAudioDevices(EDataFlow dataFlow) {
    HRESULT hr;
    CoInitialize(NULL);

    // Create a device enumerator
    IMMDeviceEnumerator* pDeviceEnumerator = NULL;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pDeviceEnumerator));
    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator\n";
        return;
    }

    // Enumerate audio endpoints
    IMMDeviceCollection* pCollection = NULL;
    hr = pDeviceEnumerator->EnumAudioEndpoints(dataFlow, DEVICE_STATE_ACTIVE, &pCollection);
    if (FAILED(hr)) {
        std::cerr << "Failed to enumerate audio endpoints\n";
        pDeviceEnumerator->Release();
        return;
    }

    UINT count;
    pCollection->GetCount(&count);

    for (UINT i = 0; i < count; ++i) {
        IMMDevice* pDevice = NULL;
        hr = pCollection->Item(i, &pDevice);
        if (SUCCEEDED(hr)) {
            // Get the device's friendly name
            IPropertyStore* pProps = NULL;
            hr = pDevice->OpenPropertyStore(STGM_READ, &pProps);
            if (SUCCEEDED(hr)) {
                PROPVARIANT varName;
                PropVariantInit(&varName);

                hr = pProps->GetValue(PKEY_Device_FriendlyName, &varName);
                if (SUCCEEDED(hr)) {
                    std::wcout << L"Device " << i << L": " << varName.pwszVal << L"\n";
                    PropVariantClear(&varName);
                }
                pProps->Release();
            }
            pDevice->Release();
        }
    }

    pCollection->Release();
    pDeviceEnumerator->Release();
    CoUninitialize();
}

int main() {
    std::cout << "Playback Devices:\n";
    ListAudioDevices(eRender); // List playback devices

    std::cout << "\nRecording Devices:\n";
    ListAudioDevices(eCapture); // List recording devices

    // record audio
    RecordAudio();

    return 0;
}
