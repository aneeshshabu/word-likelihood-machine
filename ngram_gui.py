#!/usr/bin/env python3
"""
ngram_gui.py - a window for everything in ngram.py.

Keep this file in the same folder as ngram.py, then run:
    python ngram_gui.py
(Rename it to ngram_gui.pyw to start it without a console window on Windows.)
"""
import base64
import binascii
import os
import queue
import random
import re
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
IS_WIN = sys.platform == "win32"

try:
    import ngram
    import ngram_clean
    import grammatizator as gz
except ImportError as e:
    _r = tk.Tk()
    _r.withdraw()
    messagebox.showerror(
        "Missing package",
        f"{e}\n\nKeep ngram.py, ngram_clean.py and grammatizator.py in the same folder as this file.\n"
        f"If a package is missing, open a terminal and run:\n\n\"{sys.executable}\" -m pip install numpy pypdf")
    sys.exit(1)

if IS_WIN:      # sharp text on high-DPI screens
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

MONO = ("Consolas", 10) if IS_WIN else ("DejaVu Sans Mono", 10)
UI = "Segoe UI" if IS_WIN else "DejaVu Sans"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WIN else 0


def python_exe():
    """python.exe rather than pythonw.exe, so the build process can print its progress."""
    exe = sys.executable
    if IS_WIN and exe.lower().endswith("pythonw.exe"):
        alt = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(alt):
            return alt
    return exe


def scrolled_text(parent, **kw):
    frame = ttk.Frame(parent)
    kw.setdefault("wrap", "word")
    txt = tk.Text(frame, undo=True, **kw)
    sb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    return frame, txt


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Word Likelihood Machine")
        self.geometry("980x700")
        self.minsize(760, 520)
        self.model = None
        self.model_loaded_path = None
        self.proc = None
        self.log_q = queue.Queue()

        style = ttk.Style(self)
        if IS_WIN and "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Big.TButton", padding=(10, 6))
        style.configure("Sugg.TButton", padding=(12, 8), font=(UI, 11))

        # --- top bar: which model file ------------------------------------
        top = ttk.Frame(self, padding=(10, 8, 10, 0))
        top.pack(fill="x")
        ttk.Label(top, text="Model file:").pack(side="left")
        self.model_path = tk.StringVar(value=os.path.join(HERE, "model.db"))
        ttk.Entry(top, textvariable=self.model_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Browse…", command=self.browse_model).pack(side="left")
        ttk.Button(top, text="Load", command=lambda: self.get_model(force=True)).pack(side="left", padx=(6, 0))

        self.model_info = tk.StringVar(value="No model loaded yet. Build one in the Build tab, or browse to a model.db.")
        ttk.Label(self, textvariable=self.model_info, padding=(10, 4), foreground="#555").pack(fill="x")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.nb = nb
        self._build_tab(nb)
        self._lookup_tab(nb)
        self._predict_tab(nb)
        self._generate_tab(nb)
        self._grammar_tab(nb)
        self._map_tab(nb)
        self._compress_tab(nb)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.poll_log)
        if os.path.exists(self.model_path.get()):
            self.after(200, lambda: self.get_model(quiet=True))

    # ------------------------------------------------------------------ model
    def browse_model(self):
        p = filedialog.askopenfilename(title="Choose a model",
                                       filetypes=[("Model", "*.db"), ("All files", "*.*")])
        if p:
            self.model_path.set(p)
            self.get_model(force=True)

    def unload_model(self):
        if self.model is not None:
            try:
                self.model.con.close()
            except Exception:
                pass
        self.model = None
        self.model_loaded_path = None

    def get_model(self, force=False, quiet=False):
        path = self.model_path.get().strip()
        if self.model is not None and not force and path == self.model_loaded_path:
            return self.model
        self.unload_model()
        if not os.path.exists(path):
            if not quiet:
                messagebox.showinfo("No model", f"There's no model at:\n{path}\n\n"
                                    "Build one first (Build tab) or browse to a model.db.")
            return None
        try:
            m = ngram.NGramModel(path)
        except (Exception, SystemExit) as e:
            if not quiet:
                messagebox.showerror("Couldn't open model", str(e))
            return None
        self.model, self.model_loaded_path = m, path
        meta = m.meta
        self.model_info.set(
            f"Loaded: {os.path.basename(path)}  ·  levels 1–{m.levels}  ·  "
            f"{len(m.words) - 2:,} words  ·  built from {int(meta.get('tokens', 0)):,} tokens "
            f"in {meta.get('files', '?')} file(s)  ·  {meta.get('built', '')}")
        return m

    # ------------------------------------------------------------------ Build
    def _build_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Build  ")

        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Text to learn from (.txt, .md, .pdf files or whole folders):").pack(anchor="w")
        lf = ttk.Frame(left)
        lf.pack(fill="both", expand=True, pady=4)
        self.inputs = tk.Listbox(lf, selectmode="extended", height=7, activestyle="none")
        sb = ttk.Scrollbar(lf, command=self.inputs.yview)
        self.inputs.configure(yscrollcommand=sb.set)
        self.inputs.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        bf = ttk.Frame(left)
        bf.pack(fill="x")
        ttk.Button(bf, text="Add files…", command=self.add_files).pack(side="left")
        ttk.Button(bf, text="Add folder…", command=self.add_folder).pack(side="left", padx=6)
        ttk.Button(bf, text="Remove selected",
                   command=lambda: [self.inputs.delete(i) for i in reversed(self.inputs.curselection())]
                   ).pack(side="left")
        ttk.Button(bf, text="Clear", command=lambda: self.inputs.delete(0, "end")).pack(side="left", padx=6)

        cf = ttk.LabelFrame(left, text="Cleaning (removes junk before learning)", padding=(10, 6))
        cf.pack(fill="x", pady=(10, 0))
        self.clean_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text="Clean junk first", variable=self.clean_on).grid(row=0, column=0, sticky="w")
        ttk.Label(cf, text="Text type:").grid(row=0, column=1, padx=(14, 4))
        self.clean_mode = tk.StringVar(value="auto")
        ttk.Combobox(cf, textvariable=self.clean_mode, values=["auto", "forum", "book"], state="readonly",
                     width=7).grid(row=0, column=2)
        self.grammar_filter = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text="Drop sentences with grammar errors",
                        variable=self.grammar_filter).grid(row=0, column=3, padx=(14, 0), sticky="w")
        ttk.Button(cf, text="Preview cleaning…", command=self.preview_clean).grid(row=0, column=4, padx=(14, 0))
        ttk.Label(cf, text="Blocklist file (optional):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.blocklist = tk.StringVar()
        ttk.Entry(cf, textvariable=self.blocklist).grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Button(cf, text="…", width=3, command=lambda: self.blocklist.set(
            filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All files", "*.*")]) or self.blocklist.get())
                   ).grid(row=1, column=4, sticky="w", pady=(6, 0), padx=(14, 0))
        ttk.Label(cf, foreground="#777", wraplength=620,
                  text="auto: .csv = forum posts (HTML, >>reply links, quotes, links…); .txt/.pdf = book (captions, "
                       "page headers, tables, figure references…). Blocklist: one word per line; sentences "
                       "containing them are dropped.").grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))
        cf.columnconfigure(3, weight=1)

        ttk.Label(left, text="Progress:").pack(anchor="w", pady=(10, 0))
        frame, self.log = scrolled_text(left, height=12, font=MONO, wrap="none")
        self.log.configure(state="disabled")
        frame.pack(fill="both", expand=True, pady=4)

        # options column
        right = ttk.LabelFrame(tab, text="Settings", padding=10)
        right.pack(side="right", fill="y", padx=(10, 0))
        self.out_path = tk.StringVar(value=os.path.join(HERE, "model.db"))
        ttk.Label(right, text="Save model as:").grid(row=0, column=0, columnspan=2, sticky="w")
        of = ttk.Frame(right)
        of.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Entry(of, textvariable=self.out_path, width=28).pack(side="left", fill="x", expand=True)
        ttk.Button(of, text="…", width=3, command=self.choose_output).pack(side="left", padx=(4, 0))

        self.opt = {}
        rows = [
            ("levels", "Highest level", 4, 1, 8, "4 = patterns up to 5 words"),
            ("min_count", "Min pattern count", 2, 1, 1000, "drop patterns seen fewer times"),
            ("top_k", "Next-words per context", 50, 1, 10000, "keep only the top N"),
            ("min_word_count", "Min word count", 2, 1, 1000, "rarer words become <unk>"),
            ("workers", "CPU threads", os.cpu_count() or 1, 1, 256, "processes to use"),
        ]
        r = 2
        for key, label, default, lo, hi, hint in rows:
            ttk.Label(right, text=label).grid(row=r, column=0, sticky="w", pady=2)
            var = tk.IntVar(value=default)
            ttk.Spinbox(right, from_=lo, to=hi, textvariable=var, width=7).grid(row=r, column=1, sticky="e")
            ttk.Label(right, text=hint, foreground="#777").grid(row=r + 1, column=0, columnspan=2, sticky="w")
            self.opt[key] = var
            r += 2

        ttk.Label(right, text="Count on").grid(row=r, column=0, sticky="w", pady=(6, 2))
        self.device = tk.StringVar(value="auto")
        ttk.Combobox(right, textvariable=self.device, values=["auto", "cpu", "cuda"],
                     state="readonly", width=7).grid(row=r, column=1, sticky="e", pady=(6, 2))
        r += 1
        self.gpu_status = tk.StringVar(value="auto = GPU if PyTorch can see one")
        ttk.Label(right, textvariable=self.gpu_status, foreground="#777", wraplength=230).grid(
            row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Button(right, text="Check GPU", command=self.check_gpu).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        r += 1
        self.build_btn = ttk.Button(right, text="▶  Build model", style="Big.TButton", command=self.start_build)
        self.build_btn.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        self.stop_btn = ttk.Button(right, text="■  Stop", command=self.stop_build, state="disabled")
        self.stop_btn.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        r += 1
        self.progress = ttk.Progressbar(right, mode="indeterminate")
        self.progress.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def add_files(self):
        for p in filedialog.askopenfilenames(
                title="Choose text files",
                filetypes=[("Text and PDF", "*.txt *.md *.pdf *.csv *.srt *.log"), ("All files", "*.*")]):
            self.inputs.insert("end", p)

    def add_folder(self):
        p = filedialog.askdirectory(title="Choose a folder of texts")
        if p:
            self.inputs.insert("end", p)

    def choose_output(self):
        p = filedialog.asksaveasfilename(title="Save model as", defaultextension=".db",
                                         filetypes=[("Model", "*.db")],
                                         initialfile=os.path.basename(self.out_path.get()))
        if p:
            self.out_path.set(p)

    def log_write(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s)
        self.log.see("end")
        self.log.configure(state="disabled")

    def check_gpu(self):
        self.gpu_status.set("Checking (importing PyTorch can take a few seconds)…")
        code = ("import torch;print('GPU found: '+torch.cuda.get_device_name(0) if torch.cuda.is_available() "
                "else 'PyTorch is installed but sees no CUDA GPU. Counting will use the CPU.')")

        def work():
            try:
                out = subprocess.run([python_exe(), "-c", code], capture_output=True, text=True,
                                     timeout=120, creationflags=NO_WINDOW)
                msg = out.stdout.strip() or ("PyTorch is not installed, so counting uses the CPU. "
                                             "See the README for how to install it.")
            except Exception as e:
                msg = f"Check failed: {e}"
            self.after(0, lambda: self.gpu_status.set(msg))

        threading.Thread(target=work, daemon=True).start()

    def preview_clean(self):
        items = list(self.inputs.get(0, "end"))
        sel = self.inputs.curselection()
        if not items:
            messagebox.showinfo("Preview", "Add a file first.")
            return
        path = items[sel[0]] if sel else items[0]
        if os.path.isdir(path):
            found = [f for f in ngram.collect_files([path])]
            if not found:
                messagebox.showinfo("Preview", "No .txt/.csv/.pdf files in that folder.")
                return
            path = str(found[0])
        mode, grammar = self.clean_mode.get(), self.grammar_filter.get()
        try:
            bl = frozenset(ngram_clean.load_blocklist(self.blocklist.get().strip() or None))
        except OSError:
            bl = frozenset()
        win = tk.Toplevel(self)
        win.title(f"Cleaning preview · {os.path.basename(path)}")
        win.geometry("1000x700")
        frame, txt = scrolled_text(win, font=MONO)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        txt.tag_configure("h", font=(UI, 11, "bold"))
        txt.tag_configure("before", foreground="#888")
        txt.tag_configure("after", foreground="#0a7d32")
        txt.insert("end", "Reading the file… (a big PDF or CSV can take a few seconds)")
        ttk.Button(win, text="Show other samples", command=lambda: run()).pack(pady=(0, 8))

        def run():
            def work():
                try:
                    pairs = ngram_clean.preview(path, mode, n=6, seed=random.randrange(10 ** 6),
                                                grammar=grammar, blocklist=bl)
                except Exception as e:
                    pairs = [(f"Couldn't preview: {e}", "")]
                self.after(0, lambda: show(pairs))
            threading.Thread(target=work, daemon=True).start()

        def show(pairs):
            if not win.winfo_exists():
                return
            txt.delete("1.0", "end")
            for before, after in pairs:
                txt.insert("end", "BEFORE\n", "h")
                txt.insert("end", before[:2500] + "\n\n", "before")
                txt.insert("end", "AFTER\n", "h")
                txt.insert("end", (after or "(nothing kept)") + "\n\n" + "─" * 100 + "\n\n", "after")
        run()

    def start_build(self):
        inputs = list(self.inputs.get(0, "end"))
        if not inputs:
            messagebox.showinfo("Nothing to learn from", "Add some text files or a folder first.")
            return
        out = self.out_path.get().strip()
        try:
            opts = {k: int(v.get()) for k, v in self.opt.items()}
        except (tk.TclError, ValueError):
            messagebox.showerror("Settings", "All settings must be whole numbers.")
            return
        if os.path.exists(out) and not messagebox.askyesno("Overwrite?", f"{out}\nalready exists. Replace it?"):
            return
        self.unload_model()       # Windows can't replace a file that is still open
        cmd = [python_exe(), "-u", os.path.join(HERE, "ngram.py"), "build", *inputs, "-o", out,
               "--levels", str(opts["levels"]), "--min-count", str(opts["min_count"]),
               "--top-k", str(opts["top_k"]), "--min-word-count", str(opts["min_word_count"]),
               "--workers", str(opts["workers"]), "--device", self.device.get()]
        if self.clean_on.get():
            cmd += ["--clean", "--clean-mode", self.clean_mode.get()]
            if self.grammar_filter.get():
                cmd.append("--grammar-filter")
            if self.blocklist.get().strip():
                cmd += ["--blocklist", self.blocklist.get().strip()]
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1, cwd=HERE, env=env,
                                         creationflags=NO_WINDOW)
        except Exception as e:
            messagebox.showerror("Couldn't start", str(e))
            return
        self.build_out = out
        self.build_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(12)

        def reader(proc):
            for line in proc.stdout:
                self.log_q.put(line)
            self.log_q.put(("DONE", proc.wait()))

        threading.Thread(target=reader, args=(self.proc,), daemon=True).start()

    def stop_build(self):
        if self.proc and self.proc.poll() is None:
            if IS_WIN:   # also stop the worker processes
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               capture_output=True, creationflags=NO_WINDOW)
            else:
                self.proc.terminate()
            self.log_q.put("\nStopped.\n")

    def poll_log(self):
        try:
            while True:
                item = self.log_q.get_nowait()
                if isinstance(item, tuple):
                    self.build_finished(item[1])
                else:
                    self.log_write(item)
        except queue.Empty:
            pass
        self.after(100, self.poll_log)

    def build_finished(self, code):
        self.proc = None
        self.progress.stop()
        self.build_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if code == 0:
            self.log_write("\n✔ Done. The model is loaded; try the other tabs.\n")
            self.model_path.set(self.build_out)
            self.get_model(force=True)
        else:
            self.log_write(f"\n✘ Build ended with an error (code {code}). See the messages above.\n")

    # ----------------------------------------------------------------- Lookup
    def _lookup_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Look up  ")
        f = ttk.Frame(tab)
        f.pack(fill="x")
        ttk.Label(f, text="Words:").grid(row=0, column=0, sticky="w")
        self.lk_words = tk.StringVar(value="how")
        e = ttk.Entry(f, textvariable=self.lk_words, width=40)
        e.grid(row=0, column=1, sticky="ew", padx=6)
        e.bind("<Return>", lambda _: self.do_lookup())
        ttk.Label(f, text="Check these next words (optional):").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.lk_next = tk.StringVar(value="are may")
        e2 = ttk.Entry(f, textvariable=self.lk_next, width=20)
        e2.grid(row=0, column=3, sticky="ew", padx=6)
        e2.bind("<Return>", lambda _: self.do_lookup())
        ttk.Label(f, text="Show top").grid(row=0, column=4)
        self.lk_k = tk.IntVar(value=15)
        ttk.Spinbox(f, from_=1, to=500, textvariable=self.lk_k, width=5).grid(row=0, column=5, padx=4)
        ttk.Button(f, text="Look up", style="Big.TButton", command=self.do_lookup).grid(row=0, column=6, padx=(8, 0))
        f.columnconfigure(1, weight=1)

        ttk.Label(tab, foreground="#777",
                  text="P = how likely this word comes next after the context. Each level's list adds up to 1 "
                       "(before pruning).").pack(anchor="w", pady=(8, 4))
        tf = ttk.Frame(tab)
        tf.pack(fill="both", expand=True)
        self.lk_tree = ttk.Treeview(tf, columns=("p", "count", "bar"), show="tree headings")
        self.lk_tree.heading("#0", text="Next word")
        self.lk_tree.heading("p", text="Likelihood P")
        self.lk_tree.heading("count", text="Times seen")
        self.lk_tree.heading("bar", text="")
        self.lk_tree.column("#0", width=330)
        self.lk_tree.column("p", width=110, anchor="e")
        self.lk_tree.column("count", width=110, anchor="e")
        self.lk_tree.column("bar", width=260)
        self.lk_tree.tag_configure("level", font=(UI, 10, "bold"))
        self.lk_tree.tag_configure("checked", background="#fff4c2")
        self.lk_tree.tag_configure("missing", foreground="#999")
        sb = ttk.Scrollbar(tf, command=self.lk_tree.yview)
        self.lk_tree.configure(yscrollcommand=sb.set)
        self.lk_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def do_lookup(self):
        m = self.get_model()
        if not m:
            return
        t = self.lk_tree
        t.delete(*t.get_children())
        history = ngram.tokenize(self.lk_words.get())
        if not history:
            return
        checks = [ngram.norm(w) for w in self.lk_next.get().split()]
        k = max(1, int(self.lk_k.get() or 15))
        for level in range(min(m.levels, len(history)), 0, -1):
            ctx = " ".join(history[-level:])
            rows = m.nexts(level, ctx)
            top = rows[0][2] if rows else 1
            node = t.insert("", "end", open=True, tags=("level",),
                            text=f"Level {level}  ·  after “{ctx}”  ·  {len(rows)} stored")
            for w, c, p in rows[:k]:
                tags = ("checked",) if w in checks else ()
                t.insert(node, "end", text=f"   {w}", values=(f"{p:.4f}", f"{c:,}", "█" * max(1, round(30 * p / top))),
                         tags=tags)
            for w in checks:
                hit = [(c, p) for nw, c, p in rows if nw == w]
                if hit and w not in [r[0] for r in rows[:k]]:
                    c, p = hit[0]
                    t.insert(node, "end", text=f"   {w}", values=(f"{p:.5f}", f"{c:,}", ""), tags=("checked",))
                elif not hit:
                    t.insert(node, "end", text=f"   {w}  (not stored: too rare here)",
                             values=("—", "—", ""), tags=("missing",))
            if not rows:
                t.insert(node, "end", text="   (this context never appeared in the text)", tags=("missing",))
        w = history[-1]
        if w in m.word_id:
            c = m.counts[m.word_id[w]]
            t.insert("", "end", tags=("level",),
                     text=f"Level 0  ·  “{w}” on its own", values=(f"{c / m.total:.6f}", f"{c:,}", ""))
        else:
            t.insert("", "end", tags=("level",), text=f"Level 0  ·  “{w}” is not in the vocabulary")

    # ---------------------------------------------------------------- Predict
    def _predict_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Predict (keyboard)  ")
        ttk.Label(tab, foreground="#555",
                  text="Type below. Suggestions update as you type. Click one, or press Tab to take the first."
                  ).pack(anchor="w")
        frame, self.ptext = scrolled_text(tab, height=8, font=(UI, 13))
        frame.pack(fill="x", pady=6)
        self.ptext.bind("<KeyRelease>", lambda e: self.update_predictions())
        self.ptext.bind("<ButtonRelease-1>", lambda e: self.update_predictions())
        self.ptext.bind("<Tab>", self.accept_first)

        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=4)
        self.sugg_btns = []
        for i in range(5):
            b = ttk.Button(bar, text="", style="Sugg.TButton", command=lambda i=i: self.accept(i))
            b.pack(side="left", fill="x", expand=True, padx=2)
            self.sugg_btns.append(b)
        self.suggestions = []

        opts = ttk.Frame(tab)
        opts.pack(fill="x", pady=(6, 2))
        ttk.Label(opts, text="Details (why each word was suggested):").pack(side="left")
        self.pred_grammar = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Grammar-check suggestions", variable=self.pred_grammar,
                        command=self.update_predictions).pack(side="left", padx=16)
        ttk.Button(opts, text="Clear text", command=lambda: (self.ptext.delete("1.0", "end"),
                                                             self.update_predictions())).pack(side="right")
        tf = ttk.Frame(tab)
        tf.pack(fill="both", expand=True)
        self.pred_tree = ttk.Treeview(tf, columns=("score", "src"), show="headings", height=8)
        self.pred_tree.heading("score", text="Score")
        self.pred_tree.heading("src", text="Found at")
        self.pred_tree.column("score", width=110, anchor="e")
        self.pred_tree.column("src", width=600)
        self.pred_tree["displaycolumns"] = ("score", "src")
        self.pred_tree.configure(show="tree headings")
        self.pred_tree.heading("#0", text="Word")
        self.pred_tree.column("#0", width=180)
        self.pred_tree.pack(fill="both", expand=True)

    def _split_for_prediction(self):
        before = self.ptext.get("1.0", "insert")
        m = re.search(r"[\w'’]+$", before)
        if m:
            return ngram.tokenize(before[:m.start()]), ngram.norm(m.group()), m.group()
        return ngram.tokenize(before), "", ""

    def update_predictions(self):
        m = self.get_model(quiet=True)
        if not m:
            for b in self.sugg_btns:
                b.configure(text="(load a model first)")
            return
        history, prefix, _ = self._split_for_prediction()
        preds = m.predict(history, k=30 if self.pred_grammar.get() else 10, prefix=prefix)
        blocked = []
        if self.pred_grammar.get():          # the grammatizator vetoes bad next words
            state = gz.GrammarState(m.word_id)
            for t in history:
                state.push(t)
            ok = []
            for w, sc in preds:
                errs = [i for i in state.issues_for(w, check_warnings=False) if i.severity == "error"]
                if gz.is_junk(w):
                    blocked.append((w, "number/symbol"))
                elif errs:
                    blocked.append((w, errs[0].message))
                else:
                    ok.append((w, sc))
            preds = ok[:10]
        self.suggestions = [w for w, _ in preds]
        for i, b in enumerate(self.sugg_btns):
            b.configure(text=self.suggestions[i] if i < len(self.suggestions) else "")
        t = self.pred_tree
        t.delete(*t.get_children())
        for w, (score, level) in preds:
            src = (f"level {level}: after “{' '.join(history[-level:])}”" if level
                   else "level 0: just a common word")
            t.insert("", "end", text=w, values=(f"{score:.4f}", src))
        for w, why in blocked[:4]:
            t.insert("", "end", text=w, values=("blocked", f"grammatizator: {why}"), tags=("blocked",))
        t.tag_configure("blocked", foreground="#b00")

    def accept_first(self, _event):
        if self.suggestions:
            self.accept(0)
        return "break"           # don't insert a tab character

    def accept(self, i):
        if i >= len(self.suggestions):
            return
        word = self.suggestions[i]
        _, _, typed = self._split_for_prediction()
        if typed:
            self.ptext.delete(f"insert-{len(typed)}c", "insert")
            if typed[:1].isupper():
                word = word[:1].upper() + word[1:]
        before = self.ptext.get("1.0", "insert")
        if word == "i":
            word = "I"
        elif not before.strip() or re.search(r"[.!?]\s*$", before):
            word = word[:1].upper() + word[1:]     # start of a sentence
        if not re.match(r"\w", word) and before.endswith(" "):
            self.ptext.delete("insert-1c", "insert")  # "word ," -> "word,"
        self.ptext.insert("insert", word + " ")
        self.ptext.see("insert")
        self.ptext.focus_set()
        self.update_predictions()

    # --------------------------------------------------------------- Generate
    def _generate_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Generate  ")
        f = ttk.Frame(tab)
        f.pack(fill="x")
        ttk.Label(f, text="Start with:").grid(row=0, column=0, sticky="w")
        self.gen_start = tk.StringVar(value="it was")
        e = ttk.Entry(f, textvariable=self.gen_start)
        e.grid(row=0, column=1, columnspan=5, sticky="ew", padx=6)
        e.bind("<Return>", lambda _: self.do_generate())
        ttk.Label(f, text="Words:").grid(row=1, column=0, sticky="w", pady=8)
        self.gen_n = tk.IntVar(value=80)
        ttk.Spinbox(f, from_=1, to=5000, textvariable=self.gen_n, width=6).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(f, text="Creativity (temperature):").grid(row=1, column=2, sticky="e", padx=(16, 4))
        self.gen_temp = tk.DoubleVar(value=0.8)
        self.temp_lbl = ttk.Label(f, text="0.80", width=5)
        ttk.Scale(f, from_=0.2, to=2.0, variable=self.gen_temp, length=200,
                  command=lambda v: self.temp_lbl.configure(text=f"{float(v):.2f}")).grid(row=1, column=3)
        self.temp_lbl.grid(row=1, column=4, sticky="w")
        ttk.Label(f, text="Seed (blank = random):").grid(row=1, column=5, sticky="e", padx=(16, 4))
        self.gen_seed = tk.StringVar()
        ttk.Entry(f, textvariable=self.gen_seed, width=8).grid(row=1, column=6)
        f.columnconfigure(1, weight=1)
        bf = ttk.Frame(tab)
        bf.pack(fill="x")
        ttk.Button(bf, text="✨  Generate", style="Big.TButton", command=self.do_generate).pack(side="left")
        ttk.Button(bf, text="Copy", command=lambda: self.copy(self.gen_out.get("1.0", "end-1c"))).pack(side="left", padx=6)
        self.gen_grammar = tk.BooleanVar(value=True)
        ttk.Checkbutton(bf, text="Grammatizator (rules veto bad words)", variable=self.gen_grammar).pack(side="left", padx=(16, 0))
        self.gen_nojunk = tk.BooleanVar(value=True)
        ttk.Checkbutton(bf, text="skip numbers & symbols", variable=self.gen_nojunk).pack(side="left", padx=(10, 0))
        ttk.Label(tab, foreground="#777",
                  text="Low temperature plays it safe; high temperature picks rarer words.").pack(anchor="w", pady=(6, 0))
        frame, self.gen_out = scrolled_text(tab, height=10, font=("Georgia", 12))
        frame.pack(fill="both", expand=True, pady=(8, 0))
        self.veto_lbl = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self.veto_lbl).pack(anchor="w", pady=(8, 2))
        frame, self.veto_out = scrolled_text(tab, height=5, font=MONO)
        frame.pack(fill="x")

    def do_generate(self):
        m = self.get_model()
        if not m:
            return
        seed = self.gen_seed.get().strip()
        rng = random.Random(int(seed) if seed.lstrip("-").isdigit() else None)
        self.veto_out.delete("1.0", "end")
        if self.gen_grammar.get():
            toks, vetoes = gz.generate(m, self.gen_start.get(), max(1, int(self.gen_n.get() or 80)),
                                       max(0.05, float(self.gen_temp.get())), rng, self.gen_nojunk.get())
            self.gen_out.delete("1.0", "end")
            self.gen_out.insert("1.0", ngram.detokenize(toks))
            self.veto_lbl.set(f"The grammatizator vetoed {len(vetoes)} word choice(s) the model wanted to make:")
            self.veto_out.insert("1.0", "\n".join(f"after “{c}”  ✘ {w!r:<14} {why}" for c, w, why in vetoes)
                                 or "(none: every word the model picked passed the rules)")
            return
        self.veto_lbl.set("Grammatizator off.")
        history = ngram.tokenize(self.gen_start.get())
        out = list(history)
        temp = max(0.05, float(self.gen_temp.get()))
        for _ in range(max(1, int(self.gen_n.get() or 80))):
            w = m.sample_next(history, temp, rng)
            out.append(w)
            history.append(w)
        self.gen_out.delete("1.0", "end")
        self.gen_out.insert("1.0", ngram.detokenize(out))

    # ---------------------------------------------------------------- Grammar
    def _grammar_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Grammar  ")
        ttk.Label(tab, foreground="#555",
                  text="Paste or type any text. Problems are underlined as you type: red = error, orange = warning. "
                       "Double-click a problem to find it.").pack(anchor="w")
        frame, self.gtext = scrolled_text(tab, height=9, font=(UI, 12))
        frame.pack(fill="x", pady=6)
        self.gtext.tag_configure("err", underline=True, foreground="#c00", background="#fde8e8")
        self.gtext.tag_configure("warn", underline=True, background="#fff4c2")
        self.gtext.tag_configure("pick", background="#bfe0ff")
        self.gtext.insert("1.0", "your a genius, but he dont know nothing. i has went to a apple store "
                                 "and it were more bigger then the the old one")
        self.gtext.bind("<KeyRelease>", lambda e: self._schedule_check())

        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Button(bar, text="Check", style="Big.TButton", command=self.do_check).pack(side="left")
        ttk.Button(bar, text="Fix all", command=self.do_fix).pack(side="left", padx=6)
        self.g_warn = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="show warnings", variable=self.g_warn, command=self.do_check).pack(side="left", padx=10)
        ttk.Button(bar, text="All rules…", command=self.show_rules).pack(side="left", padx=6)
        self.g_status = tk.StringVar()
        ttk.Label(bar, textvariable=self.g_status, font=(UI, 11, "bold")).pack(side="left", padx=14)

        tf = ttk.Frame(tab)
        tf.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("where", "kind", "problem", "fix", "rule")
        self.g_tree = ttk.Treeview(tf, columns=cols, show="headings")
        for c, w, txt in zip(cols, (70, 70, 420, 140, 150), ("Where", "Kind", "Problem", "Suggested fix", "Rule")):
            self.g_tree.heading(c, text=txt)
            self.g_tree.column(c, width=w, anchor="w")
        self.g_tree.tag_configure("error", foreground="#c00")
        self.g_tree.tag_configure("warning", foreground="#a65c00")
        sb = ttk.Scrollbar(tf, command=self.g_tree.yview)
        self.g_tree.configure(yscrollcommand=sb.set)
        self.g_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.g_tree.bind("<Double-1>", self._goto_issue)
        self._g_job = None
        self._g_issues = []
        self.after(300, self.do_check)

    def _schedule_check(self):
        if self._g_job:
            self.after_cancel(self._g_job)
        self._g_job = self.after(400, self.do_check)

    def _vocab(self):
        return self.model.word_id if self.model is not None else None

    def do_check(self):
        text = self.gtext.get("1.0", "end-1c")
        issues = gz.check(text, self._vocab(), self.g_warn.get())
        self._g_issues = issues
        for tag in ("err", "warn", "pick"):
            self.gtext.tag_remove(tag, "1.0", "end")
        self.g_tree.delete(*self.g_tree.get_children())
        for k, iss in enumerate(issues):
            a, b = f"1.0+{iss.start}c", f"1.0+{iss.end}c"
            self.gtext.tag_add("err" if iss.severity == "error" else "warn", a, b)
            line = text.count("\n", 0, iss.start) + 1
            col = iss.start - (text.rfind("\n", 0, iss.start) + 1) + 1
            fix = iss.fix if iss.fix else ("(delete)" if iss.fix == "" else "")
            self.g_tree.insert("", "end", iid=str(k), tags=(iss.severity,),
                               values=(f"{line}:{col}", iss.severity, f"“{text[iss.start:iss.end]}”: {iss.message}",
                                       fix, iss.rule))
        n_err = sum(i.severity == "error" for i in issues)
        n_warn = len(issues) - n_err
        self.g_status.set("✔ No problems found." if not issues else f"{n_err} error(s), {n_warn} warning(s)")

    def do_fix(self):
        text = self.gtext.get("1.0", "end-1c")
        fixed = gz.fix_text(text, self._vocab(), self.g_warn.get())
        if fixed != text:
            self.gtext.delete("1.0", "end")
            self.gtext.insert("1.0", fixed)
        self.do_check()

    def _goto_issue(self, _e):
        sel = self.g_tree.selection()
        if not sel:
            return
        iss = self._g_issues[int(sel[0])]
        self.gtext.tag_remove("pick", "1.0", "end")
        a, b = f"1.0+{iss.start}c", f"1.0+{iss.end}c"
        self.gtext.tag_add("pick", a, b)
        self.gtext.see(a)

    def show_rules(self):
        win = tk.Toplevel(self)
        win.title(f"The grammatizator's {len(gz.RULES)} rules")
        win.geometry("900x640")
        frame, txt = scrolled_text(win, font=(UI, 11))
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        txt.tag_configure("t", font=(UI, 12, "bold"))
        txt.tag_configure("bad", foreground="#c00")
        txt.tag_configure("good", foreground="#0a7d32")
        txt.tag_configure("dim", foreground="#777")
        for rid, sev, title, explain, bad, good, _ in gz.RULES:
            txt.insert("end", f"{title}", "t")
            txt.insert("end", f"   {sev} · {rid}\n", "dim")
            txt.insert("end", explain + "\n")
            txt.insert("end", f"   ✘ {bad}\n", "bad")
            txt.insert("end", f"   ✔ {good}\n\n", "good")
        txt.insert("end", f"Word lists: {len(gz.IRR)} irregular verbs with all their forms, {len(gz.BASE_VERBS)} verbs "
                          f"checked for he/she/it agreement, {len(gz.MISSPELLED)} common misspellings, plus articles, "
                          f"pronouns, modals and the forms of be/have/do.", "dim")
        txt.configure(state="disabled")

    # --------------------------------------------------------------- Word map
    def _map_tab(self, nb):
        tab = ttk.Frame(nb, padding=14)
        nb.add(tab, text="  Word map  ")
        ttk.Label(tab, text="3D map of how the words in the loaded model connect", font=(UI, 13, "bold")).pack(anchor="w")
        ttk.Label(tab, foreground="#555", wraplength=900,
                  text="Makes a web page from the model shown at the top of this window and opens it in your browser. "
                       "Ball size = how common a word is · colour = words that stick together · line = often side by "
                       "side · click a word to see what comes before and after it.").pack(anchor="w", pady=(4, 12))
        f = ttk.LabelFrame(tab, text="Settings", padding=12)
        f.pack(fill="x")
        self.map_vars = {}
        rows = [("words", "Words on the map", 500, 50, 1500, "more words = takes longer to arrange"),
                ("links", "Links per word", 5, 1, 15, "how many of each word's strongest connections to draw"),
                ("skip", "Hide most common words", 0, 0, 300, "e.g. 20 hides 'the', 'of', 'and' (they link to everything)")]
        for r, (key, label, default, lo, hi, hint) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=4)
            var = tk.IntVar(value=default)
            ttk.Spinbox(f, from_=lo, to=hi, textvariable=var, width=7).grid(row=r, column=1, sticky="w", padx=8)
            ttk.Label(f, text=hint, foreground="#777").grid(row=r, column=2, sticky="w")
            self.map_vars[key] = var
        self.map_punct = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Include punctuation (. , ! ?) as words", variable=self.map_punct).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        bf = ttk.Frame(tab)
        bf.pack(fill="x", pady=14)
        self.map_btn = ttk.Button(bf, text="🌐  Make map and open it", style="Big.TButton", command=self.make_map)
        self.map_btn.pack(side="left")
        self.map_open_btn = ttk.Button(bf, text="Open last map", state="disabled",
                                       command=lambda: webbrowser.open(self.last_map.as_uri()))
        self.map_open_btn.pack(side="left", padx=8)
        ttk.Button(bf, text="Show in folder", command=self.show_map_folder).pack(side="left")
        self.map_bar = ttk.Progressbar(tab, mode="indeterminate")
        self.map_bar.pack(fill="x")
        self.map_status = tk.StringVar(value="Load a model at the top, then click Make map.")
        ttk.Label(tab, textvariable=self.map_status, font=(UI, 11)).pack(anchor="w", pady=(8, 0))
        ttk.Label(tab, foreground="#777", wraplength=900,
                  text="The first map downloads the 3D library once (about 0.6 MB) and keeps it in a map_assets "
                       "folder. After that maps work offline, and each map is one .html file you can share.").pack(
            anchor="w", pady=(16, 0))
        self.last_map = None

    def make_map(self):
        path = self.model_path.get().strip()
        if not os.path.exists(path):
            messagebox.showinfo("Word map", "Load a model first (top of the window), or build one in the Build tab.")
            return
        try:
            import ngram_map
        except ImportError as e:
            messagebox.showerror("Word map", f"{e}\n\nKeep ngram_map.py in the same folder as this file.")
            return
        try:
            opts = dict(words=int(self.map_vars["words"].get()), links=int(self.map_vars["links"].get()),
                        skip_top=int(self.map_vars["skip"].get()), punct=self.map_punct.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Word map", "The settings must be whole numbers.")
            return
        self.map_btn.configure(state="disabled")
        self.map_bar.start(12)

        def work():
            try:
                out, n, e = ngram_map.make_map(path, progress=lambda m: self.after(0, self.map_status.set, m), **opts)
                self.after(0, self.map_done, out, None)
            except Exception as ex:
                self.after(0, self.map_done, None, ex)
        threading.Thread(target=work, daemon=True).start()

    def map_done(self, out, err):
        self.map_bar.stop()
        self.map_bar.configure(value=0)
        self.map_btn.configure(state="normal")
        if err:
            self.map_status.set(f"Problem: {err}")
            messagebox.showerror("Word map", str(err))
            return
        self.last_map = out
        self.map_open_btn.configure(state="normal")
        self.map_status.set(self.map_status.get() + f"  →  opened in your browser ({out})")
        webbrowser.open(out.as_uri())

    def show_map_folder(self):
        target = self.last_map or Path(self.model_path.get()).parent
        if IS_WIN:
            if self.last_map:
                subprocess.run(["explorer", "/select,", str(self.last_map)])
            else:
                os.startfile(str(target))
        else:
            webbrowser.open(Path(target if not self.last_map else self.last_map.parent).as_uri())

    # --------------------------------------------------------------- Compress
    def _compress_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Compress  ")
        ttk.Label(tab, foreground="#555",
                  text="Both sides need the same model file. The code is plain text, so you can paste it anywhere."
                  ).pack(anchor="w")
        panes = ttk.Panedwindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True, pady=6)

        top = ttk.Frame(panes)
        hdr = ttk.Frame(top)
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Message:").pack(side="left")
        ttk.Button(hdr, text="Open text file…", command=self.open_text_file).pack(side="right")
        frame, self.cmp_in = scrolled_text(top, height=8, font=(UI, 11))
        frame.pack(fill="both", expand=True, pady=4)
        panes.add(top, weight=1)

        mid = ttk.Frame(panes)
        bf = ttk.Frame(mid)
        bf.pack(fill="x", pady=4)
        ttk.Button(bf, text="⬇  Compress", style="Big.TButton", command=self.do_compress).pack(side="left")
        ttk.Button(bf, text="⬆  Decompress", style="Big.TButton", command=self.do_decompress).pack(side="left", padx=6)
        self.cmp_stats = tk.StringVar()
        ttk.Label(bf, textvariable=self.cmp_stats, font=MONO).pack(side="left", padx=10)
        hdr2 = ttk.Frame(mid)
        hdr2.pack(fill="x")
        ttk.Label(hdr2, text="Compressed code:").pack(side="left")
        ttk.Button(hdr2, text="Paste", command=self.paste_code).pack(side="right")
        ttk.Button(hdr2, text="Copy", command=lambda: self.copy(self.cmp_out.get("1.0", "end-1c"))).pack(side="right", padx=6)
        frame, self.cmp_out = scrolled_text(mid, height=5, font=MONO, wrap="char")
        frame.pack(fill="both", expand=True, pady=4)
        panes.add(mid, weight=1)

    def open_text_file(self):
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt *.md"), ("All files", "*.*")])
        if p:
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                self.cmp_in.delete("1.0", "end")
                self.cmp_in.insert("1.0", f.read())

    def do_compress(self):
        m = self.get_model()
        if not m:
            return
        text = self.cmp_in.get("1.0", "end-1c")
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            data = ngram.compress_text(m, text)
            ok = ngram.decompress_bytes(m, data) == text
        except (Exception, SystemExit) as e:
            messagebox.showerror("Compress failed", str(e))
            return
        finally:
            self.config(cursor="")
        if not ok:
            messagebox.showerror("Compress failed", "Round-trip check failed. Please tell Claude!")
            return
        code = base64.urlsafe_b64encode(data).decode().rstrip("=")
        self.cmp_out.delete("1.0", "end")
        self.cmp_out.insert("1.0", code)
        raw = len(text.encode("utf-8"))
        z = len(ngram._deflate(text.encode("utf-8")))
        pct = 100 * len(data) / max(raw, 1)
        self.cmp_stats.set(f"{raw:,} bytes → zip {z:,} → n-gram {len(data):,} bytes ({pct:.0f}%)  ·  "
                           f"code: {len(code):,} characters")

    def do_decompress(self):
        m = self.get_model()
        if not m:
            return
        code = re.sub(r"\s+", "", self.cmp_out.get("1.0", "end-1c"))
        if not code:
            messagebox.showinfo("Decompress", "Paste a code into the 'Compressed code' box first.")
            return
        try:
            data = base64.urlsafe_b64decode(code + "=" * (-len(code) % 4))
            text = ngram.decompress_bytes(m, data)
        except SystemExit as e:
            messagebox.showerror("Can't decompress", str(e))
            return
        except (binascii.Error, ValueError, IndexError, StopIteration, UnicodeDecodeError, Exception):
            messagebox.showerror("Can't decompress", "That code is damaged or incomplete.")
            return
        self.cmp_in.delete("1.0", "end")
        self.cmp_in.insert("1.0", text)
        self.cmp_stats.set(f"Decompressed {len(data):,} bytes → {len(text.encode('utf-8')):,} bytes")

    def paste_code(self):
        try:
            s = self.clipboard_get()
        except tk.TclError:
            return
        self.cmp_out.delete("1.0", "end")
        self.cmp_out.insert("1.0", s)

    # ------------------------------------------------------------------ misc
    def copy(self, s):
        self.clipboard_clear()
        self.clipboard_append(s)

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Build running", "A build is still running. Stop it and quit?"):
                return
            self.stop_build()
        self.unload_model()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
