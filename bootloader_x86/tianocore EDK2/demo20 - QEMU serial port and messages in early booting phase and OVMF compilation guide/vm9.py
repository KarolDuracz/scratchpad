#!/usr/bin/env python3

import sys, math, traceback, time, threading, queue
from collections import deque, Counter
from capstone import Cs, CS_ARCH_X86, CS_MODE_32
from capstone.x86_const import X86_OP_MEM, X86_OP_REG, X86_OP_IMM
from unicorn import Uc, UC_ARCH_X86, UC_MODE_32, UC_PROT_READ, UC_PROT_WRITE, UC_PROT_EXEC
from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE, UC_HOOK_MEM_FETCH
from unicorn import UC_HOOK_MEM_READ_UNMAPPED, UC_HOOK_MEM_WRITE_UNMAPPED, UC_HOOK_MEM_FETCH_UNMAPPED
import unicorn.x86_const as uc_x86_const
from unicorn.x86_const import *
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk

# ---- Config ----
PAGE_SIZE = 0x1000
PHYS_TOP = 0x100000000
RESET_VECTOR = 0xFFFFFFF0
MAX_INSTRUCTIONS = 3_000_000
STACK_BASE = 0x00080000
STACK_SIZE = 0x20000

LOW_RAM_MAP_SIZE = 0x01000000   # 16MB
DEFAULT_RAM_BASE = 0x00800000
DEFAULT_RAM_SIZE = 8 * 1024 * 1024

AUTO_MAP_WRITES = True     # if False, write-to-unmapped will be recorded but not auto-mapped
AUTO_MAP_READS = True      # if False, read-from-unmapped will be diagnostic (not recommended)
RECENT_INSNS = 2000
LOOP_DETECT_WINDOW = 4096
LOOP_DETECT_COUNT = 1500

GUI_MAX_W = 1200
GUI_MAX_H = 800
GUI_UPDATE_MS = 40

REG_NAME_TO_UC = {
    'eax': UC_X86_REG_EAX, 'ebx': UC_X86_REG_EBX, 'ecx': UC_X86_REG_ECX,
    'edx': UC_X86_REG_EDX, 'esi': UC_X86_REG_ESI, 'edi': UC_X86_REG_EDI,
    'ebp': UC_X86_REG_EBP, 'esp': UC_X86_REG_ESP, 'eip': UC_X86_REG_EIP,
    'cs': UC_X86_REG_CS, 'ds': UC_X86_REG_DS, 'es': UC_X86_REG_ES,
    'fs': UC_X86_REG_FS, 'gs': UC_X86_REG_GS, 'ss': UC_X86_REG_SS,
}

def page_down(addr): return addr & ~(PAGE_SIZE - 1)
def page_up(n): return (n + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)

# ---- Visualizer (page -> single pixel mapping) ----
class Visualizer:
    def __init__(self, tracked_regions, mmio_ranges, fw_map_base, fw_size):
        self.tracked_regions = tracked_regions
        self.mmio_ranges = mmio_ranges
        self.fw_map_base = fw_map_base
        self.fw_size = fw_size

        pages = {}
        for base, size, label in self.tracked_regions:
            start = page_down(base); end = page_up(base + size)
            for p in range(start, end, PAGE_SIZE):
                pages[p] = label
        if not pages:
            start = page_down(self.fw_map_base); end = page_up(self.fw_map_base + self.fw_size)
            for p in range(start, end, PAGE_SIZE): pages[p] = 'FW'
        self.page_addrs = sorted(pages.keys())
        self.page_labels = pages
        self.page_count = len(self.page_addrs)

        self.max_pixels = GUI_MAX_W * GUI_MAX_H
        self.pages_per_pixel = max(1, math.ceil(self.page_count / self.max_pixels))
        self.pixels_needed = math.ceil(self.page_count / self.pages_per_pixel)
        self.canvas_w = min(GUI_MAX_W, max(64, int(math.sqrt(self.pixels_needed))))
        self.canvas_h = max(1, math.ceil(self.pixels_needed / self.canvas_w))
        if self.canvas_w * self.canvas_h < self.pixels_needed:
            self.canvas_h = math.ceil(self.pixels_needed / self.canvas_w)

        self.pixel_colors = [(20,20,20)] * (self.canvas_w * self.canvas_h)
        self.reset_to_mapped_state()

        self.q = queue.Queue()
        self.root = None
        self.img_label = None
        self.tkimg = None
        self.text_regs = None
        self.status_var = None

    def reset_to_mapped_state(self):
        for i in range(len(self.pixel_colors)):
            self.pixel_colors[i] = (20,20,20)
        # color MMIO pages
        for p in self.page_addrs:
            for mmio_base, mmio_sz in self.mmio_ranges:
                if mmio_base <= p < mmio_base + mmio_sz:
                    idx = self._page_to_pixel_index(p)
                    if idx is not None:
                        self.pixel_colors[idx] = (90,0,90)
        # color firmware pages
        start = page_down(self.fw_map_base); end = page_up(self.fw_map_base + self.fw_size)
        for p in range(start, end, PAGE_SIZE):
            idx = self._page_to_pixel_index(p)
            if idx is not None and self.pixel_colors[idx] == (20,20,20):
                self.pixel_colors[idx] = (0,40,0)

    def _page_to_pixel_index(self, page_addr):
        # binary-search style improvement would be faster; using simple index() is okay for tens of thousands
        try:
            idx = self.page_addrs.index(page_addr)
        except ValueError:
            return None
        pixel = idx // self.pages_per_pixel
        if pixel < 0 or pixel >= len(self.pixel_colors):
            return None
        return pixel

    def mark_event(self, addr, access_type):
        try:
            self.q.put_nowait(('mem', addr, access_type))
        except Exception:
            pass

    def push_regs(self, regs_snapshot):
        try:
            self.q.put_nowait(('regs', regs_snapshot))
        except Exception:
            pass

    def create_window(self):
        self.root = tk.Tk()
        self.root.wm_title("UC VM Memory Visualizer (page->pixel)")
        self.img = Image.new('RGB', (self.canvas_w, self.canvas_h))
        for i, c in enumerate(self.pixel_colors):
            x = i % self.canvas_w; y = i // self.canvas_w
            self.img.putpixel((x,y), c)
        display_w = min(self.canvas_w, GUI_MAX_W//2)
        display_h = min(self.canvas_h, GUI_MAX_H//2)
        self.tkimg = ImageTk.PhotoImage(self.img.resize((max(300, display_w), max(200, display_h)), Image.NEAREST))
        self.img_label = tk.Label(self.root, image=self.tkimg)
        self.img_label.grid(row=0, column=0, sticky='nsew')
        self.text_regs = tk.Text(self.root, width=48, height=40, bg='black', fg='lime', font=('Consolas',10))
        self.text_regs.grid(row=0, column=1, sticky='nsew')
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var).grid(row=1, column=0, columnspan=2, sticky='we')
        self.root.after(GUI_UPDATE_MS, self._periodic_update)

    def _periodic_update(self):
        changed = False
        regs_update = None
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            try:
                if ev[0] == 'mem':
                    _, addr, atype = ev
                    page = page_down(addr)
                    idx = self._page_to_pixel_index(page)
                    if idx is None:
                        continue
                    if atype == 'w': col = (255,0,0)
                    elif atype == 'r': col = (0,120,255)
                    elif atype == 'x': col = (0,220,0)
                    else: col = (100,100,100)
                    self.pixel_colors[idx] = col
                    changed = True
                elif ev[0] == 'regs':
                    regs_update = ev[1]
            except Exception:
                continue

        if changed:
            img = Image.new('RGB', (self.canvas_w, self.canvas_h))
            px = img.load()
            for y in range(self.canvas_h):
                for x in range(self.canvas_w):
                    pidx = y * self.canvas_w + x
                    c = self.pixel_colors[pidx] if pidx < len(self.pixel_colors) else (0,0,0)
                    px[x,y] = c
            display_w = max(300, min(GUI_MAX_W//2, self.canvas_w*2))
            display_h = max(200, min(GUI_MAX_H//2, self.canvas_h*2))
            self.tkimg = ImageTk.PhotoImage(img.resize((display_w, display_h), Image.NEAREST))
            self.img_label.configure(image=self.tkimg)
            self.img_label.image = self.tkimg

        if regs_update is not None:
            try:
                self.text_regs.delete('1.0', tk.END)
                lines = []
                for k,v in sorted(regs_update.items()):
                    lines.append(f"{k:6s} = {('0x%08X' % v) if isinstance(v,int) else v}")
                self.text_regs.insert(tk.END, "\n".join(lines))
            except Exception:
                pass

        self.root.after(GUI_UPDATE_MS, self._periodic_update)

    def start_mainloop(self):
        self.root.mainloop()

# ---- helpers ----
def map_ram_region(mu, base, size):
    try:
        mu.mem_map(page_down(base), page_up(size), UC_PROT_READ | UC_PROT_WRITE)
        try:
            mu.mem_write(page_down(base), b'\x00' * page_up(size))
        except Exception:
            pass
        return True
    except Exception:
        try:
            mu.mem_read(page_down(base), 1)
            return True
        except Exception:
            return False

def ensure_pages_mapped_for_write(mu, addr, size):
    start = page_down(addr); end = page_up(addr + size); length = end - start
    try:
        mu.mem_map(start, length, UC_PROT_READ | UC_PROT_WRITE | UC_PROT_EXEC)
        return True
    except Exception:
        try:
            mu.mem_map(start, PAGE_SIZE, UC_PROT_READ | UC_PROT_WRITE | UC_PROT_EXEC)
            return True
        except Exception:
            return False

def ensure_pages_mapped_and_zero(mu, addr, size):
    start = page_down(addr); end = page_up(addr + size); length = end - start
    try:
        mu.mem_map(start, length, UC_PROT_READ | UC_PROT_WRITE | UC_PROT_EXEC)
    except Exception:
        try:
            mu.mem_map(start, PAGE_SIZE, UC_PROT_READ | UC_PROT_WRITE | UC_PROT_EXEC)
        except Exception:
            return False
    try:
        mu.mem_write(start, b'\x00' * length)
    except Exception:
        pass
    return True

def is_mapped(mu, addr):
    try:
        mu.mem_read(page_down(addr), 1)
        return True
    except Exception:
        return False

# ---- runner ----
def run_firmware_with_vis(fd_path):
    fw = open(fd_path, 'rb').read()
    fw_size = len(fw)
    print(f"Firmware: {fd_path}, size={fw_size} bytes")

    uc = Uc(UC_ARCH_X86, UC_MODE_32)

    # pre-map low RAM
    try:
        uc.mem_map(0x0, LOW_RAM_MAP_SIZE, UC_PROT_READ | UC_PROT_WRITE | UC_PROT_EXEC)
        uc.mem_write(0x0, b'\x00' * LOW_RAM_MAP_SIZE)
        print(f"[init] mapped low RAM 0x0 - 0x{LOW_RAM_MAP_SIZE:08X}")
    except Exception as e:
        print("[init] failed to map low RAM:", e)

    # map firmware high
    desired_map_base = PHYS_TOP - fw_size
    anchor = page_down(desired_map_base)
    offset = desired_map_base - anchor
    map_size = page_up(offset + fw_size)
    uc.mem_map(anchor, map_size, UC_PROT_READ | UC_PROT_WRITE | UC_PROT_EXEC)
    uc.mem_write(desired_map_base, fw)
    fw_map_base = desired_map_base
    print(f"Mapped firmware -> phys 0x{fw_map_base:08X}")

    # stack
    try:
        uc.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
    except Exception:
        pass
    initial_esp = STACK_BASE + STACK_SIZE - 0x10
    uc.reg_write(UC_X86_REG_ESP, initial_esp)
    print(f"Stack at 0x{STACK_BASE:08X} initial ESP=0x{initial_esp:08X}")

    # pragmatic RAM
    if not map_ram_region(uc, DEFAULT_RAM_BASE, DEFAULT_RAM_SIZE):
        print("[init] default RAM mapping failed")
    else:
        print(f"[init] mapped pragmatic RAM at 0x{DEFAULT_RAM_BASE:08X} size=0x{DEFAULT_RAM_SIZE:X}")

    # MMIO stubs
    MMIO_RANGES = [(0xFEC00000, PAGE_SIZE), (0xFEE00000, PAGE_SIZE), (0xFED00000, PAGE_SIZE), (0xFED80000, PAGE_SIZE)]
    for b,s in MMIO_RANGES:
        try:
            st = page_down(b); su = page_up(s)
            uc.mem_map(st, su, UC_PROT_READ | UC_PROT_WRITE)
            uc.mem_write(b, b'\x00' * s)
            print(f"[init] MMIO mapped 0x{b:08X}")
        except Exception:
            pass

    # init regs
    uc.reg_write(UC_X86_REG_EAX, 0); uc.reg_write(UC_X86_REG_EBX, 0)
    uc.reg_write(UC_X86_REG_ECX, 0); uc.reg_write(UC_X86_REG_EDX, 0)
    uc.reg_write(UC_X86_REG_EBP, 0); uc.reg_write(UC_X86_REG_ESI, 0)
    uc.reg_write(UC_X86_REG_EDI, 0); uc.reg_write(UC_X86_REG_EFLAGS, 0x2)
    uc.reg_write(UC_X86_REG_EIP, RESET_VECTOR & 0xFFFFFFFF)

    cs = Cs(CS_ARCH_X86, CS_MODE_32); cs.detail = True

    tracked = [(0x0, LOW_RAM_MAP_SIZE, 'LOWRAM'), (fw_map_base, fw_size, 'FW'),
               (DEFAULT_RAM_BASE, DEFAULT_RAM_SIZE, 'RAM'), (STACK_BASE, STACK_SIZE, 'STACK')]

    vis = Visualizer(tracked, MMIO_RANGES, fw_map_base, fw_size)
    vis.create_window()

    recent_insns = deque(maxlen=RECENT_INSNS)
    recent_rips = deque(maxlen=LOOP_DETECT_WINDOW)
    instr_count = 0
    diagnostics = []
    pci_cfg = {'addr':0}

    def snapshot_regs(mu):
        d={}
        for n,r in REG_NAME_TO_UC.items():
            try: d[n.upper()] = mu.reg_read(r) & 0xffffffff
            except Exception: d[n.upper()] = None
        try: d['EFLAGS'] = mu.reg_read(UC_X86_REG_EFLAGS) & 0xffffffff
        except Exception: d['EFLAGS'] = None
        # add other MSR-like placeholders if you want
        return d

    # Unmapped memory handlers: map pages on demand (safe fallback)
    def hook_unmapped(mu, access, address, size, value, user_data):
        try:
            # decide read/write/fetch
            if access == UC_MEM_FETCH:
                kind = 'x'
            elif access == UC_MEM_READ:
                kind = 'r'
            elif access == UC_MEM_WRITE:
                kind = 'w'
            else:
                kind = '?'
            # try mapping the page(s)
            start = page_down(address)
            if user_data and isinstance(user_data, dict) and user_data.get('allow_map') is False:
                # configured to not map -> stop emu
                vis.status_var.set(f"Unmapped access @0x{address:08X}")
                mu.emu_stop()
                return
            ok = ensure_pages_mapped_and_zero(mu, address, size) if (access != UC_MEM_WRITE or AUTO_MAP_WRITES) else ensure_pages_mapped_for_write(mu, address, size)
            vis.mark_event(address, kind)
            if not ok:
                vis.status_var.set(f"cannot map page @0x{address:08X}")
                # stop emulation if mapping failed
                try: mu.emu_stop()
                except Exception: pass
        except Exception:
            # ensure exceptions in callback are swallowed and don't corrupt ctypes conversion
            traceback.print_exc()

    # install unmapped hooks (these will auto-map pages on demand)
    try:
        uc.hook_add(UC_HOOK_MEM_FETCH_UNMAPPED, hook_unmapped)
        uc.hook_add(UC_HOOK_MEM_READ_UNMAPPED, hook_unmapped)
        uc.hook_add(UC_HOOK_MEM_WRITE_UNMAPPED, hook_unmapped)
    except Exception as e:
        print("Warning: could not add UNMAPPED hooks:", e)

    # instrumentation hooks (fire visualizer events)
    def hook_mem_read(mu, access, address, size, value, user_data):
        try:
            vis.mark_event(address, 'r')
        except Exception:
            pass
    def hook_mem_write(mu, access, address, size, value, user_data):
        try:
            vis.mark_event(address, 'w')
        except Exception:
            pass
    def hook_mem_fetch(mu, access, address, size, value, user_data):
        try:
            vis.mark_event(address, 'x')
        except Exception:
            pass

    try:
        uc.hook_add(UC_HOOK_MEM_READ, hook_mem_read)
        uc.hook_add(UC_HOOK_MEM_WRITE, hook_mem_write)
        uc.hook_add(UC_HOOK_MEM_FETCH, hook_mem_fetch)
    except Exception:
        pass

    # helpers to parse a single instruction at EIP
    def read_single_insn(mu):
        try:
            ip = mu.reg_read(UC_X86_REG_EIP) & 0xffffffff
            raw = mu.mem_read(ip, 16)
            for insn in cs.disasm(raw, ip, count=1):
                return insn
        except Exception:
            return None
        return None

    # handle cpuid, in, out in code hook to avoid UC_HOOK_INSN ctypes fragility
    def emulate_cpuid(mu, insn):
        try:
            eax_in = mu.reg_read(UC_X86_REG_EAX) & 0xffffffff
        except Exception:
            eax_in = 0
        # small conservative stub: respond with zeros except for requested leaf 80000000
        try:
            if eax_in == 0x80000000:
                mu.reg_write(UC_X86_REG_EAX, 0x8000001F)
                mu.reg_write(UC_X86_REG_EBX,0); mu.reg_write(UC_X86_REG_ECX,0); mu.reg_write(UC_X86_REG_EDX,0)
            else:
                mu.reg_write(UC_X86_REG_EAX,0); mu.reg_write(UC_X86_REG_EBX,0); mu.reg_write(UC_X86_REG_ECX,0); mu.reg_write(UC_X86_REG_EDX,0)
            diagnostics.append({'evt':'cpuid_emulated','leaf':hex(eax_in)})
        except Exception:
            pass

    def emulate_io_in(mu, insn):
        # very small port stub: support PCI config ports CF8/CFC as before
        try:
            port = None
            size = 4
            for op in insn.operands:
                if op.type == X86_OP_IMM:
                    port = op.imm & 0xffff
                elif op.type == X86_OP_REG:
                    if cs.reg_name(op.reg).lower() == 'dx':
                        port = mu.reg_read(UC_X86_REG_EDX) & 0xffff
            if port is None:
                return
            # handle CFC (pci config data)
            if port == 0xCFC:
                mu.reg_write(UC_X86_REG_EAX, 0)
                diagnostics.append({'evt':'pci_cfg_read','port':hex(port)})
            else:
                # generic input zero
                mu.reg_write(UC_X86_REG_EAX, 0)
                diagnostics.append({'evt':'io_in','port':hex(port)})
        except Exception:
            pass

    def emulate_io_out(mu, insn):
        try:
            port = None
            for op in insn.operands:
                if op.type == X86_OP_IMM:
                    port = op.imm & 0xffff
                elif op.type == X86_OP_REG:
                    if cs.reg_name(op.reg).lower() == 'dx':
                        port = mu.reg_read(UC_X86_REG_EDX) & 0xffff
            if port is None:
                return
            if port == 0xCF8:
                try:
                    val = mu.reg_read(UC_X86_REG_EAX) & 0xffffffff
                except Exception:
                    val = 0
                pci_cfg['addr'] = val
                diagnostics.append({'evt':'pci_cfg_addr_write','val':hex(val)})
            else:
                diagnostics.append({'evt':'io_out','port':hex(port)})
        except Exception:
            pass

    # main code hook
    def hook_code(mu, address, size, user_data):
        nonlocal instr_count
        instr_count += 1
        try:
            try:
                raw = mu.mem_read(address, size)
            except Exception:
                # fallback to firmware image if fetch inside firmware mapping we wrote earlier
                if fw_map_base <= address < fw_map_base + fw_size:
                    off = address - fw_map_base
                    raw = fw[off:off+size]
                    if len(raw) < size:
                        raw = raw + b'\x90' * (size - len(raw))
                else:
                    vis.status_var.set(f"fetch fail @0x{address:08X}")
                    mu.emu_stop(); return
            # decode single instruction
            for insn in cs.disasm(raw, address, count=1):
                b = raw[:insn.size]; bhex = ' '.join(f"{x:02X}" for x in b)
                mnem = insn.mnemonic.lower()

                recent_insns.append((instr_count, insn.address, bhex, insn.mnemonic, insn.op_str, insn.size))
                recent_rips.append(insn.address)

                # hot-loop detection & conservative step-out
                if len(recent_rips) >= LOOP_DETECT_WINDOW:
                    c = Counter(recent_rips); top, cnt = c.most_common(1)[0]
                    if cnt >= LOOP_DETECT_COUNT:
                        print(f"Hot loop at 0x{top:08X} count={cnt}. Stepping out.")
                        try:
                            cur_eip = mu.reg_read(UC_X86_REG_EIP) & 0xffffffff
                            cur_raw = mu.mem_read(cur_eip, 16)
                            for ci in cs.disasm(cur_raw, cur_eip, count=1):
                                step = max(1, ci.size)
                                new_eip = (cur_eip + step * 8) & 0xffffffff
                                # ensure page mapped before writing EIP
                                if not is_mapped(mu, new_eip):
                                    ensure_pages_mapped_and_zero(mu, new_eip, 4)
                                mu.reg_write(UC_X86_REG_EIP, new_eip)
                                diagnostics.append({'evt': 'hot_loop_patch','from':cur_eip,'to':new_eip})
                                print(f"[hot-loop] moved EIP from 0x{cur_eip:08X} to 0x{new_eip:08X}")
                                break
                        except Exception as e:
                            print("[hot-loop] step-out failed:", e)
                        recent_rips.clear()

                # mark execute page
                vis.mark_event(insn.address, 'x')
                
                """
                # occasional push of regs to GUI
                if instr_count % 1000 == 0:
                    try:
                        regs = snapshot_regs(mu)
                        vis.push_regs(regs)
                    except Exception:
                        pass
                """
                
                # push regs to GUI every instruction (REAL-TIME; very chatty)
                try:
                    regs = snapshot_regs(mu)
                    vis.push_regs(regs)
                except Exception:
                    pass
                
                # emulate privileged/IO insns inside hook to avoid UC_HOOK_INSN
                try:
                    if mnem == 'cpuid':
                        emulate_cpuid(mu, insn)
                    elif mnem.startswith('in'):
                        emulate_io_in(mu, insn)
                    elif mnem.startswith('out'):
                        emulate_io_out(mu, insn)
                except Exception:
                    pass

                # analyze memory operands heuristically and auto-map if needed
                try:
                    for idx, op in enumerate(insn.operands):
                        if op.type == X86_OP_MEM:
                            opm = op.mem
                            ea = None
                            if opm.base == 0 and opm.index == 0:
                                ea = opm.disp & 0xffffffff
                            else:
                                try:
                                    base_val = 0
                                    if opm.base != 0:
                                        nm = cs.reg_name(opm.base).lower()
                                        if nm in REG_NAME_TO_UC:
                                            base_val = mu.reg_read(REG_NAME_TO_UC[nm]) & 0xffffffff
                                    idx_val = 0
                                    if opm.index != 0:
                                        nm = cs.reg_name(opm.index).lower()
                                        if nm in REG_NAME_TO_UC:
                                            idx_val = mu.reg_read(REG_NAME_TO_UC[nm]) & 0xffffffff
                                    ea = (base_val + idx_val * opm.scale + (opm.disp & 0xffffffff)) & 0xffffffff
                                except Exception:
                                    ea = None
                            if ea is None:
                                continue
                            is_write = (idx == 0 and mnem.startswith('mov')) or mnem.startswith('stos')
                            if is_write:
                                if not is_mapped(mu, ea):
                                    if AUTO_MAP_WRITES:
                                        ensure_pages_mapped_for_write(mu, ea, getattr(op, 'size', 1) or 1)
                                    else:
                                        diagnostics.append({'evt':'write_unmapped','ea':ea,'insn':(insn.address,mnem,insn.op_str)})
                            else:
                                # read operand
                                if not is_mapped(mu, ea) and AUTO_MAP_READS:
                                    ensure_pages_mapped_and_zero(mu, ea, getattr(op,'size',1) or 1)
                except Exception:
                    pass

                """
                # small periodic console logging
                if instr_count % 2000 == 0:
                    print(f"[{instr_count}] 0x{insn.address:08X}: {bhex}  {insn.mnemonic} {insn.op_str}")
                """
                
                # print every instruction in real-time (VERY verbose; will slow emulation)
                print(f"[{instr_count}] 0x{insn.address:08X}: {bhex}  {insn.mnemonic} {insn.op_str}", flush=True)


                break
        except Exception:
            # swallow exceptions from the hook so ctypes doesn't get corrupted
            traceback.print_exc()
            try:
                vis.status_var.set("Hook exception (see console)")
            except Exception:
                pass

        if instr_count >= MAX_INSTRUCTIONS:
            try:
                mu.emu_stop()
            except Exception:
                pass

    uc.hook_add(UC_HOOK_CODE, hook_code)

    # run emulator in background thread
    def emu_thread():
        try:
            uc.emu_start(RESET_VECTOR, 0xFFFFFFFF, timeout=0, count=MAX_INSTRUCTIONS)
        except Exception as e:
            print("Emulation stopped:", type(e).__name__, e)
            try:
                vis.status_var.set(f"Emu stopped: {type(e).__name__} {e}")
            except Exception:
                pass
            traceback.print_exc()

    t = threading.Thread(target=emu_thread, daemon=True)
    t.start()

    try:
        vis.start_mainloop()
    except KeyboardInterrupt:
        print("GUI closed, stopping.")
    except Exception as e:
        print("GUI error:", e)
        traceback.print_exc()

    # short join attempt; daemon thread will exit with process
    t.join(timeout=0.1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python vm9_vis_improved.py /path/to/OVMF.fd")
        sys.exit(1)
    run_firmware_with_vis(sys.argv[1])
