// this not working - console not working
#if 1
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