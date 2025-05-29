#!/usr/bin/env python3
import os, re, sys, html
from collections import defaultdict, Counter

# --- 1) Regex patterns for C “definitions” ---
FUNC_DEF_RE = re.compile(r'^\s*([_A-Za-z]\w[\w\s\*]+?)\s+([_A-Za-z]\w*)\s*\(.*?\)\s*{')
GLOBAL_VAR_RE = re.compile(r'^\s*([_A-Za-z]\w[\w\s\*]+?)\s+([_A-Za-z]\w*)\s*;\s*(?:/\*.*)?$')
#TOKEN_RE = re.compile(r'\b[_A-Za-z]\w*\b')

TOKEN_RE = re.compile(r'\b[_A-Za-z][_A-Za-z0-9]*\b')

def build_index(rootdir):
    """
    Walk rootdir for .c/.h, record:
      all_defs[name] = [(file, lineno), ...]
      file_uses[file] = Counter(name->count)
    """
    all_defs = defaultdict(list)
    file_uses = {}
    for dirpath, _, files in os.walk(rootdir):
        for fn in files:
            if not fn.endswith(('.c','.h')):
                continue
            path = os.path.join(dirpath, fn)
            uses = Counter()

         
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for lineno, raw in enumerate(f,1):
                    # 1) strip C single‐line and multi‐line comments:
                    line = re.sub(r'//.*|/\*.*?\*/', ' ', raw)
                    # 2) strip string literals (simple double‐quoted):
                    line = re.sub(r'"(\\.|[^"\\])*"', '""', line)
                    # 1a) function defs
                    m = FUNC_DEF_RE.match(line)
                    if m:
                        name = m.group(2)
                        all_defs[name].append((path, lineno))
                    # 1b) global var defs
                    m2 = GLOBAL_VAR_RE.match(line)
                    if m2:
                        name = m2.group(2)
                        all_defs[name].append((path, lineno))

                    # 2) count uses
                    for tok in TOKEN_RE.findall(line):
                        uses[tok] += 1
            file_uses[path] = uses
    return all_defs, file_uses

def linkify_line(line, defs, srcfile, file_uses, total_uses, src_root):
    """
    Wrap tokens in <a> if in defs, with multi-site tooltip.
    """

    #src_dir = os.path.dirname(srcfile)
    src_dir = os.path.dirname(srcfile)
    # strip comments & strings here too, but keep original for HTML
    stripped = re.sub(r'//.*|/\*.*?\*/', ' ', line)
    stripped = re.sub(r'"(\\.|[^"\\])*"', '""', stripped)
    def repl(m):
        name = m.group(0)
        if name not in defs:
            return html.escape(name)
        sites = defs[name]
        # href to first def
        df0, ln0 = sites[0]
        rel = os.path.relpath(df0, start=src_dir) + f'.html#L{ln0}'
        # build tooltip
        lines = ["Defined at:"]
        for dfile, dln in sites:
            relp = os.path.relpath(dfile, start=src_root)
            lines.append(f"  {relp}:{dln}")
        here = file_uses[srcfile].get(name,0)
        total = total_uses[name]
        lines.append(f"Uses here: {here}")
        lines.append(f"Total uses: {total}")
        title = "\\n".join(lines)
        return (f'<a href="{html.escape(rel)}" title="{html.escape(title)}">'
                f'{html.escape(name)}</a>')
    #return TOKEN_RE.sub(repl, line)
    # only match tokens in the cleaned‐up version, but output based on original
    out = []
    last = 0
    for m in TOKEN_RE.finditer(stripped):
        out.append(html.escape(line[last:m.start()]))
        out.append(repl(m))
        last = m.end()
    out.append(html.escape(line[last:]))
    return ''.join(out)

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage: c_xref.py <src_root> <out_html_dir>")
        sys.exit(1)
    src_root, out_root = sys.argv[1], sys.argv[2]

    all_defs, file_uses = build_index(src_root)
    total_uses = Counter()
    for u in file_uses.values():
        total_uses.update(u)

    for srcfile, uses in file_uses.items():
        rel = os.path.relpath(srcfile, src_root)
        outpath = os.path.join(out_root, rel + '.html')
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        with open(srcfile,'r',encoding='utf-8',errors='ignore') as sf, \
             open(outpath,'w',encoding='utf-8') as out:

            # --- header + CSS + panels ---
            out.write(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{rel}</title>
<style>
html,body {{ height:100%; margin:0; }}
body {{
  display:flex; white-space:pre; font-family:monospace;
}}
#main {{ flex:1 1 auto; overflow:auto; padding:10px; }}
#history {{
  flex:0 0 200px; max-width:200px;
  background:#eef; border-left:1px solid #ccc;
  padding:5px; font-size:12px;
  white-space:normal; word-wrap:break-word;
  overflow-y:auto; height:100vh;
}}
#summary {{ margin-bottom:1em; padding:.5em; background:#f0f0f0; }}
a {{ text-decoration:none; color:#c00; cursor:pointer; }}
.highlight-def {{ background:#ccf !important; }}
.highlight-use {{ background:#cfc !important; }}
</style>
</head><body>
<div id="main">
<div id="summary">
  <strong>File:</strong> {rel}<br>
  <strong>Top symbols:</strong><br>
""")
            # --- top 5 symbols ---
            for name,cnt in uses.most_common(5):
                sites = all_defs.get(name,[])
                if sites:
                    df0,ln0 = sites[0]
                    r = os.path.relpath(df0, start=os.path.dirname(srcfile))
                    href = f"{r}.html#L{ln0}"
                    lines = ["Defined at:"] + [
                        f"  {os.path.relpath(dfile,src_root)}:{dln}"
                        for dfile,dln in sites
                    ] + [
                        f"Uses here: {cnt}",
                        f"Total uses: {total_uses[name]}"
                    ]
                    title = "\\n".join(lines)
                    link = (f'<a href="{html.escape(href)}" '
                            f'title="{html.escape(title)}">{html.escape(name)}</a>')
                else:
                    link = html.escape(name)
                out.write(f"    {link} : {cnt}<br>\n")
            out.write("</div>\n")

            # --- emit source with links & line anchors ---
            for ln,raw in enumerate(sf,1):
                txt = raw.rstrip('\n')
                linked = linkify_line(txt, all_defs, srcfile, file_uses, total_uses, src_root)
                out.write(f'<a id="L{ln}"></a>{ln:4d}: {linked}\n')

            # --- close main, history panel & JS ---
            out.write("""
</div>
<div id="history"><strong>Click history:</strong><br></div>
<script>
document.addEventListener("DOMContentLoaded",()=>{
  const main=document.getElementById("main");
  const hist=document.getElementById("history");
  main.querySelectorAll("a").forEach(el=>{
    el.addEventListener("click",e=>{
      const name=el.textContent, title=el.title||"";
      // add to history
      const d=document.createElement("div");
      d.innerHTML="<b>"+name+"</b><br>"+title.replace(/\\n/g,"<br>");
      hist.appendChild(d); hist.scrollTop=hist.scrollHeight;
      // clear highlights
      document.querySelectorAll(".highlight-def").forEach(x=>x.classList.remove("highlight-def"));
      document.querySelectorAll(".highlight-use").forEach(x=>x.classList.remove("highlight-use"));
      // highlight tokens
      Array.from(main.querySelectorAll("a"))
           .filter(a=>a.textContent===name)
           .forEach(a=>{
             if(a.href.endsWith("#L"+a.id.slice(1))) a.classList.add("highlight-def");
             else a.classList.add("highlight-use");
           });
      e.preventDefault();
    });
  });
});
</script>
</body></html>
""")
    print("C cross-ref HTML generated in", out_root)
