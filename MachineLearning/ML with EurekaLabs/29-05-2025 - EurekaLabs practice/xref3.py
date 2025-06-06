#!/usr/bin/env python3
import ast, os, re, sys, html
from collections import defaultdict, Counter

class SymbolIndexer(ast.NodeVisitor):
    def __init__(self, filename, defs, uses):
        self.filename = filename
        self.defs = defs      # shared: name → list of (file, lineno)
        self.uses = uses      # per-file Counter: name → count

    def visit_FunctionDef(self, node):
        self.defs[node.name].append((self.filename, node.lineno))
        for arg in node.args.args:
            self.defs[arg.arg].append((self.filename, node.lineno))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.defs[node.name].append((self.filename, node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defs[target.id].append((self.filename, target.lineno))
        self.generic_visit(node)

    def visit_Name(self, node):
        self.uses[node.id] += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.uses[node.func.id] += 1
        self.generic_visit(node)

def build_index(rootdir):
    all_defs = defaultdict(list)
    file_uses = {}
    for dirpath, _, filenames in os.walk(rootdir):
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                src = open(path, 'r', encoding='utf-8').read()
                tree = ast.parse(src, path)
            except (SyntaxError, UnicodeDecodeError):
                continue
            uses = Counter()
            SymbolIndexer(path, all_defs, uses).visit(tree)
            file_uses[path] = uses
    return all_defs, file_uses

TOKEN_RE = re.compile(r'\b[_A-Za-z]\w*\b')

def linkify_line(line, defs, srcfile, file_uses, total_uses, src_root):
    """
    Wrap every token in <a> if it’s in defs, with a 'title' tooltip listing
    all definition sites and usage stats.
    """
    srcfile_dir = os.path.dirname(srcfile)
    def replace(m):
        name = m.group(0)
        if name not in defs:
            return html.escape(name)
        def_sites = defs[name]
        # Use first definition for href:
        df0, ln0 = def_sites[0]
        rel_href = os.path.relpath(df0, start=srcfile_dir) + f'.html#L{ln0}'
        # Build tooltip text listing all sites + stats
        tooltip_lines = ["Defined at:"]
        for dfile, dln in def_sites:
            rel = os.path.relpath(dfile, start=src_root)
            tooltip_lines.append(f"  {rel}:{dln}")
        uses_here = file_uses[srcfile].get(name, 0)
        uses_total = total_uses[name]
        tooltip_lines.append(f"Uses in this file: {uses_here}")
        tooltip_lines.append(f"Total uses in project: {uses_total}")
        # Join with literal newline so most browsers render multiline title
        title = "\\n".join(tooltip_lines)
        return (f'<a href="{html.escape(rel_href)}" '
                f'title="{html.escape(title)}">'
                f'{html.escape(name)}</a>')
    return TOKEN_RE.sub(replace, line)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python xref_tooltip.py <source_root_dir> <output_html_dir>")
        sys.exit(1)

    src_root = sys.argv[1]
    out_root = sys.argv[2]

    # 1) Build definitions & per-file usage
    all_defs, file_uses = build_index(src_root)

    # 2) Compute total uses across project
    total_uses = Counter()
    for uses in file_uses.values():
        total_uses.update(uses)

    # 3) Generate HTML for each source
    for srcfile, uses in file_uses.items():
        rel_src = os.path.relpath(srcfile, src_root)
        outpath = os.path.join(out_root, rel_src + '.html')
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        with open(srcfile, 'r', encoding='utf-8') as sf, \
             open(outpath, 'w', encoding='utf-8') as out:

            out.write("""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
  <title>{rel_src}</title>
    <style>
    body {
      font-family: monospace;
      white-space: pre;
      margin: 0;
      display: flex;
    }
    #main {
      flex-grow: 1;
      padding-left: 10px;
    }
    #summary {
      padding: .5em;
      background: #f0f0f0;
    }
    history {

  /* fix the width and prevent shrinking: */
  flex: 0 0 200px;
  max-width: 200px;

  /* normal wrapping, not preformatted: */
  white-space: normal !important;
  word-wrap: break-word;
  overflow-y: auto;
  overflow-x: hidden;

  background: #eef;
  border-left: 1px solid #ccc;
  padding: 5px;
  font-size: 12px;
  height: 100vh;
}
    a {
      text-decoration: none;
      color: #c00;
      cursor: pointer;
    }
    .highlight-def {
      background: #ccf !important;
    }
    .highlight-use {
      background: #cfc !important;
    }
  </style>
</head>
<body>
<div id="main">
<div id="summary">
  <strong>File:</strong> {rel_src}<br>
  <strong>Top symbols by usage:</strong><br>

""")
 # --- your existing top-5 loop, unchanged ---
            for name, cnt in uses.most_common(5):
                def_sites = all_defs.get(name, [])
                if def_sites:
                    df0, ln0 = def_sites[0]
                    rel = os.path.relpath(df0, start=os.path.dirname(srcfile))
                    href = f"{rel}.html#L{ln0}"
                    lines = ["Defined at:"] + [
                        f"  {os.path.relpath(dfile, start=src_root)}:{dln}"
                        for dfile, dln in def_sites
                    ] + [
                        f"Uses in this file: {cnt}",
                        f"Total uses in project: {total_uses[name]}"
                    ]
                    title = "\\n".join(lines)
                    link = (f'<a href="{html.escape(href)}" '
                            f'title="{html.escape(title)}">{html.escape(name)}</a>')
                else:
                    link = html.escape(name)
                out.write(f"    {link} : {cnt}<br>\n")

            # close summary
            out.write("</div>\n")

            # annotate each source line exactly as before
            for lineno, raw in enumerate(sf, 1):
                txt = raw.rstrip('\n')
                linked = linkify_line(
                    txt, all_defs, srcfile, file_uses, total_uses, src_root
                )
                out.write(f'<a id="L{lineno}"></a>{lineno:4d}: {linked}\n')

            # --- now close #main and create #history + JS ---
            out.write("""
</div>  <!-- end #main -->
<div id="history">
  <strong>Click history:</strong><br>
</div>

<script>
document.addEventListener("DOMContentLoaded", () => {
  const main = document.getElementById("main");
  const historyDiv = document.getElementById("history");

  main.querySelectorAll("a").forEach(el => {
    el.addEventListener("click", e => {
      const name = el.textContent;
      const title = el.title || "";

      // 1) append full tooltip info
      const entry = document.createElement("div");
      entry.innerHTML = "<b>" + name + "</b><br>" +
                        title.replace(/\\n/g, "<br>");
      historyDiv.appendChild(entry);
      historyDiv.scrollTop = historyDiv.scrollHeight;

      // 2) clear old highlights
      document.querySelectorAll(".highlight-def").forEach(e => e.classList.remove("highlight-def"));
      document.querySelectorAll(".highlight-use").forEach(e => e.classList.remove("highlight-use"));

      // 3) highlight all occurrences of that token
      Array.from(main.querySelectorAll("a"))
           .filter(a => a.textContent === name)
           .forEach(a => {
             // def if its anchor matches its own href’s line
             if (a.href.endsWith("#L" + a.id.slice(1))) {
               a.classList.add("highlight-def");
             } else {
               a.classList.add("highlight-use");
             }
           });

      e.preventDefault();
    });
  });
});
</script>

</body></html>
""")

    print("HTML with full-definition tooltips and click history written to", out_root)

