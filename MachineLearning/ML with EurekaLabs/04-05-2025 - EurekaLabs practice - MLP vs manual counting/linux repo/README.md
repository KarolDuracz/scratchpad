<h2>linux repo analyze</h2>

Script to "walk through" folders to find .c files

```
PATH = "C:\\Users\\kdhome\\Documents\\progs\\EurekaLabs\\08-05-2025 - analyze linux repo\\linux-master\\linux-master"
```

python3 repo_tools.py ~/dev/linux .c all_sources.txt - Originally the code runs from the console with arguments, but here I used notebook for this and the PATH path to the main linux code folder as above.


```
#!/usr/bin/env python3
import os
import sys

def list_tree(startpath: str):
    """
    Print all directories and files under startpath in a tree-like structure.
    """
    for root, dirs, files in os.walk(startpath):
        # Compute the depth to create the tree indent
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}📄 {f}")

def collect_by_extension(startpath: str) -> dict:
    """
    Walk the directory tree and return a dict mapping file-extensions to lists of file-paths.
    Files without an extension are grouped under the key ''.
    """
    ext_map = {}
    for dirpath, _, filenames in os.walk(startpath):
        for fname in filenames:
            _, ext = os.path.splitext(fname)
            ext = ext.lower()  # normalize
            fullpath = os.path.join(dirpath, fname)
            ext_map.setdefault(ext, []).append(fullpath)
    return ext_map

def concatenate_files(file_list: list, output_path: str):
    """
    Given a list of file paths, read each in turn and append its contents
    into the output_path file, with a header between files.
    """
    with open(output_path, 'w', encoding='utf-8', errors='ignore') as out_f:
        for idx, fpath in enumerate(file_list, start=1):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as in_f:
                    out_f.write(f"/* ===== File {idx}: {fpath} ===== */\n")
                    out_f.write(in_f.read())
                    out_f.write("\n\n")
                print(f"[+] Appended: {fpath}")
            except Exception as e:
                print(f"[!] Skipped {fpath}: {e}")
"""
# Print tree and collect all .c files into all_sources.txt
python3 repo_tools.py ~/dev/linux .c all_sources.txt
"""
def main():
    """
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-repo> [extension] [output-file]")
        print("  <path-to-repo> : root directory to scan")
        print("  [extension]    : file extension to collect (e.g. .c). Default: .c")
        print("  [output-file]  : where to write combined contents. Default: combined.txt")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    ext = sys.argv[2] if len(sys.argv) > 2 else '.c'
    out_file = sys.argv[3] if len(sys.argv) > 3 else 'combined.txt'
    """

    repo_path = PATH
    ext = '.c'
    out_file = 'combined.txt'
    
    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a directory.")
        sys.exit(1)

    # 1. Print the directory tree
    print("=== Repository Tree ===")
    list_tree(repo_path)

    # 2. Build extension map
    print("\n=== Gathering files ===")
    ext_map = collect_by_extension(repo_path)
    file_list = ext_map.get(ext.lower(), [])

    if not file_list:
        print(f"No files with extension '{ext}' found.")
        sys.exit(0)

    print(f"Found {len(file_list)} '{ext}' files. Writing to '{out_file}'...")

    # 3. Concatenate them
    concatenate_files(file_list, out_file)
    print("Done.")

if __name__ == '__main__':
    main()
```

Script run from command - python script.py - from console to cut into pieces large file "combined.txt" where all source codes are copied. Each file is 50 MB.

```
# split_large_file.py

def split_file(input_file='combined.txt', max_chunk_size=50 * 1024 * 1024):
    """
    Split a large file into multiple smaller files, each approximately max_chunk_size bytes.
    Default chunk size is 50 MB.
    """
    i = 0
    size = 0

    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
            out_file = open(f'part_{i}.txt', 'w', encoding='utf-8')
            for line in infile:
                line_size = len(line.encode('utf-8'))
                if size + line_size > max_chunk_size:
                    out_file.close()
                    i += 1
                    out_file = open(f'part_{i}.txt', 'w', encoding='utf-8')
                    size = 0
                out_file.write(line)
                size += line_size
            out_file.close()
        print(f"✅ Done. Created {i + 1} part files.")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    split_file()
```
