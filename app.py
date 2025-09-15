# app.py
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
import json
import os
import threading
from pathlib import Path
import sys

# Optional deps for proving
try:
    import ezkl
    from pytictoc import TicToc
    EZKL_AVAILABLE = True
except Exception:
    EZKL_AVAILABLE = False

# App config
DISPLAY_MAX = 500
PREVIEW_W = 200
PREVIEW_H = 200
TARGET_SQ_SIZE = min(PREVIEW_W, PREVIEW_H)
RESIZED_DISPLAY_SCALE = 4
JSON_NORMALIZE = True
DEFAULT_INPUT_JSON = "input_image_rgb_64x64.json"
PROOFS_DIRNAME = "proofs"
SRS_PATH = os.path.expanduser("~/.ezkl/srs/kzg17.srs")  # local fallback; CLI flow uses base_dir/srs/kzg.srs
SRS_LOG2_DEGREE = 18

# Installation (one-time CLI setup) — returns a base dir where artifacts live
from setup import run_installation

# ---------- small utils ----------

def log_safe(widget: tk.Text | None, msg: str):
    if not widget:
        print(msg)
        return
    widget.configure(state=tk.NORMAL)
    widget.insert(tk.END, msg + "\n")
    widget.see(tk.END)
    widget.configure(state=tk.DISABLED)
    widget.update_idletasks()

def find_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def search_for(filename, start_dir):
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def autodiscover_artifacts(base_dir, logger=None):
    compiled = find_first([
        os.path.join(base_dir, "network.ezkl"),
        os.path.join(base_dir, "model.compiled"),
    ]) or search_for("network.ezkl", base_dir) or search_for("model.compiled", base_dir)

    settings = find_first([os.path.join(base_dir, "settings.json")]) or search_for("settings.json", base_dir)
    pk = find_first([os.path.join(base_dir, "pk.key")]) or search_for("pk.key", base_dir)
    vk = find_first([os.path.join(base_dir, "vk.key")]) or search_for("vk.key", base_dir)

    if logger:
        log_safe(logger, f"[auto] compiled_model: {compiled}")
        log_safe(logger, f"[auto] settings     : {settings}")
        log_safe(logger, f"[auto] pk          : {pk}")
        log_safe(logger, f"[auto] vk          : {vk}")
    return compiled, settings, pk, vk

def ensure_srs(srs_path, k, logger=None):
    if not EZKL_AVAILABLE:
        return False
    srs_dir = os.path.dirname(srs_path)
    os.makedirs(srs_dir, exist_ok=True)
    if not os.path.exists(srs_path):
        if logger: log_safe(logger, f"[ezkl] SRS not found; generating: {srs_path} (k={k})")
        ezkl.gen_srs(srs_path, k)
    else:
        if logger: log_safe(logger, f"[ezkl] SRS already present: {srs_path}")
    return True

def render_into_box(img, frame_w, frame_h, target_size, resample):
    bg = Image.new("RGB", (frame_w, frame_h), "#111")
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img_scaled = img.resize((new_w, new_h), resample)
    x = (frame_w - new_w) // 2
    y = (frame_h - new_h) // 2
    bg.paste(img_scaled, (x, y))
    return bg

def infer_and_build_proof(compiled_model_path, settings_path, pk_path, vk_path,
                          input_path, output_path, logger=None):
    if not EZKL_AVAILABLE:
        raise RuntimeError("ezkl is not installed/available in this environment.")

    witness_path = "witness.json"
    t = TicToc()

    if logger: log_safe(logger, f"[ezkl] gen_witness → {witness_path}")
    t.tic()
    _ = ezkl.gen_witness(
        input_path,
        compiled_model_path,
        witness_path,
        vk_path,
        settings_path
    )
    elapsed_gen_witness = t.tocvalue()
    if logger: log_safe(logger, f"[timing] Gen Witness: {elapsed_gen_witness:.3f}s")

    if logger: log_safe(logger, f"[ezkl] prove → {output_path}")
    t.tic()
    _ = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        output_path,
        "single",
    )
    elapsed_prove = t.tocvalue()
    if logger:
        log_safe(logger, f"[timing] Prove: {elapsed_prove:.3f}s")
        log_safe(logger, "[ezkl] Done.")

    assert os.path.isfile(output_path), f"Proof was not written to {output_path}"
    try:
        os.remove(witness_path)
    except Exception:
        pass

# ---------- GUI app ----------

class ImageToTensorApp(tk.Tk):
    def __init__(self, app_base_dir: Path):
        super().__init__()
        self.title("Image → RGB 3×64×64 + EZKL Proof")
        self.minsize(1200, 860)

        self.app_base_dir = Path(app_base_dir).resolve()

        # Keep references so Tk doesn't GC images
        self._orig_photo = None
        self._resized_photo = None

        # Data holders
        self.tensor_chw = None
        self.json_str = None
        self.input_image_path = None
        self.input_json_path = None
        self.proof_bytes = None
        self.proof_hex = None
        self.proof_result = None
        self.proof_path = None
        self.output_hex = None
        self.settings_path = None
        self.vk_path = None

        # UI layout
        top = tk.Frame(self, padx=10, pady=10); top.pack(fill=tk.X)
        tk.Button(top, text="Select Image…", command=self.select_image).pack(side=tk.LEFT)
        self.info_var = tk.StringVar(value="No image loaded")
        tk.Label(top, textvariable=self.info_var, anchor="w").pack(side=tk.LEFT, padx=12)

        body = tk.Frame(self, padx=10, pady=10); body.pack(fill=tk.BOTH, expand=True)
        left = tk.Frame(body); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="Original").pack(anchor="w")
        self.orig_frame = tk.Frame(left, width=PREVIEW_W, height=PREVIEW_H, bd=1, relief=tk.SOLID, bg="#111")
        self.orig_frame.pack(padx=(0, 8), pady=4); self.orig_frame.pack_propagate(False)
        self.orig_label = tk.Label(self.orig_frame, bg="#111"); self.orig_label.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(body); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right, text="Converted (RGB 64x64; display scaled)").pack(anchor="w")
        self.res_frame = tk.Frame(right, width=PREVIEW_W, height=PREVIEW_H, bd=1, relief=tk.SOLID, bg="#111")
        self.res_frame.pack(padx=(8, 8), pady=4); self.res_frame.pack_propagate(False)
        self.resized_label = tk.Label(self.res_frame, bg="#111"); self.resized_label.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(self, padx=10, pady=8); controls.pack(fill=tk.X)
        self.btn_gen_proof = tk.Button(controls, text="Generate Proof", command=self.generate_proof, state=tk.NORMAL)
        self.btn_gen_proof.pack(side=tk.LEFT)
        tk.Label(controls, text=f"(JSON saved in {self.app_base_dir})").pack(side=tk.LEFT, padx=8)

        outputs = tk.Frame(self, padx=10, pady=10); outputs.pack(fill=tk.BOTH, expand=False)
        proof_frame = tk.LabelFrame(outputs, text="Proof Hex", padx=10, pady=10); proof_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.proof_text = ScrolledText(proof_frame, height=10, wrap="char")
        self.proof_text.pack(fill=tk.BOTH, expand=True); self.proof_text.configure(state=tk.DISABLED)

        actions = tk.Frame(proof_frame); actions.pack(fill=tk.X, pady=(8, 0))
        self.btn_copy_proof = tk.Button(actions, text="Copy Hex", command=self.copy_proof, state=tk.DISABLED); self.btn_copy_proof.pack(side=tk.LEFT)
        self.btn_save_proof = tk.Button(actions, text="Save Proof File…", command=self.save_proof, state=tk.DISABLED); self.btn_save_proof.pack(side=tk.LEFT, padx=8)

        out_frame = tk.LabelFrame(outputs, text="Public Output (hex)", padx=10, pady=10, width=320)
        out_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0)); out_frame.pack_propagate(False)
        self.output_hex_text = ScrolledText(out_frame, height=10, wrap="char"); self.output_hex_text.pack(fill=tk.BOTH, expand=True); self.output_hex_text.configure(state=tk.DISABLED)
        self.btn_copy_output_hex = tk.Button(out_frame, text="Copy Output Hex", command=self.copy_output_hex, state=tk.DISABLED)
        self.btn_copy_output_hex.pack(pady=(8, 0), anchor="w")

        result_frame = tk.LabelFrame(outputs, text="Proof Result (rescaled_outputs)", padx=10, pady=10, width=250)
        result_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(8, 0)); result_frame.pack_propagate(False)
        self.result_var = tk.StringVar(value="—")
        self.result_label = tk.Label(result_frame, textvariable=self.result_var, anchor="center", font=("TkDefaultFont", 14))
        self.result_label.pack(fill=tk.BOTH, expand=True)

        verify_frame = tk.LabelFrame(self, text="Verification", padx=10, pady=10); verify_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.verify_text = tk.Text(verify_frame, height=3, wrap="word"); self.verify_text.pack(fill=tk.X, expand=False); self.verify_text.configure(state=tk.DISABLED)
        self.btn_verify = tk.Button(verify_frame, text="Verify Proof", command=self.verify_proof, state=tk.DISABLED); self.btn_verify.pack(anchor="w", pady=(8, 0))

        log_frame = tk.LabelFrame(self, text="Logs", padx=10, pady=10); log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = ScrolledText(log_frame, height=10, wrap="word"); self.log_text.pack(fill=tk.BOTH, expand=True); self.log_text.configure(state=tk.DISABLED)

        if not EZKL_AVAILABLE:
            log_safe(self.log_text, "[warn] ezkl / pytictoc not available. Generating a proof will show a warning.")

    # ----------------------- Main logic -----------------------
    def select_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            orig_w, orig_h = img.size

            img_rgb = img.convert("RGB")
            resized_64_rgb = img_rgb.resize((64, 64), Image.LANCZOS)

            arr_rgb_u8 = np.array(resized_64_rgb, dtype=np.uint8)
            self.tensor_chw = np.transpose(arr_rgb_u8, (2, 0, 1))

            if JSON_NORMALIZE:
                arr = self.tensor_chw.astype(np.float32) / 255.0
            else:
                arr = self.tensor_chw.astype(np.float32)
            flat = arr.reshape(-1)
            payload = {"input_data": [flat.tolist()]}
            self.json_str = json.dumps(payload, indent=2, ensure_ascii=False)
            self.input_image_path = path

            orig_canvas = render_into_box(img.convert("RGB"), PREVIEW_W, PREVIEW_H, TARGET_SQ_SIZE, Image.LANCZOS)
            self._orig_photo = ImageTk.PhotoImage(orig_canvas)
            self.orig_label.configure(image=self._orig_photo)

            right_canvas = render_into_box(resized_64_rgb, PREVIEW_W, PREVIEW_H, TARGET_SQ_SIZE, Image.NEAREST)
            self._resized_photo = ImageTk.PhotoImage(right_canvas)
            self.resized_label.configure(image=self._resized_photo)

            self.info_var.set(
                f"Original: {orig_w}×{orig_h} | Tensor: {self.tensor_chw.shape} (RGB, CHW) "
                f"| JSON floats: {flat.size} ({'normalized' if JSON_NORMALIZE else '0..255'})"
            )

            # Reset outputs
            self.proof_text.configure(state=tk.NORMAL); self.proof_text.delete("1.0", tk.END); self.proof_text.configure(state=tk.DISABLED)
            self.btn_copy_proof.config(state=tk.DISABLED); self.btn_save_proof.config(state=tk.DISABLED)
            self.proof_bytes = None; self.proof_hex = None; self.proof_result = None; self.proof_path = None
            self.result_var.set("—")
            self.output_hex = None
            self.output_hex_text.configure(state=tk.NORMAL); self.output_hex_text.delete("1.0", tk.END); self.output_hex_text.insert(tk.END, "—"); self.output_hex_text.configure(state=tk.DISABLED)
            self.btn_copy_output_hex.config(state=tk.DISABLED)

            log_safe(self.log_text, f"[app] Prepared image and JSON for: {os.path.basename(path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not open/process image:\n{e}")

    def _write_input_json_next_to_app(self):
        if not self.json_str:
            raise RuntimeError("No JSON prepared.")
        # Write into the base artifact dir (NOT CWD)
        out_path = Path(self.app_base_dir) / DEFAULT_INPUT_JSON
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.json_str)
        self.input_json_path = str(out_path)
        log_safe(self.log_text, f"[app] Wrote input JSON → {self.input_json_path}")

    def copy_output_hex(self):
        if not self.output_hex:
            messagebox.showwarning("No output", "No public output hex available to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(self.output_hex)
        self.update()
        messagebox.showinfo("Copied", "Public output hex copied to clipboard.")

    def generate_proof(self):
        if not EZKL_AVAILABLE:
            messagebox.showwarning("EZKL missing", "ezkl is not available. Install ezkl to enable proving.")
            return
        if not self.json_str:
            messagebox.showwarning("No JSON", "Load an image to create JSON first.")
            return

        try:
            self._write_input_json_next_to_app()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write input JSON:\n{e}")
            return

        base_dir = str(self.app_base_dir)
        compiled_model_path, settings_path, pk_path, vk_path = autodiscover_artifacts(base_dir, self.log_text)
        missing = [("model.compiled / network.ezkl", compiled_model_path),
                   ("settings.json", settings_path),
                   ("pk.key", pk_path),
                   ("vk.key", vk_path)]
        problems = [name for name, p in missing if not p]

        self.settings_path = settings_path
        self.vk_path = vk_path

        if problems:
            messagebox.showerror(
                "Artifacts not found",
                "Could not find the following in your app folder:\n- " + "\n- ".join(problems) +
                f"\n\nLooked under:\n{self.app_base_dir}"
            )
            return

        # Optional: python-based SRS ensure (CLI already fetched SRS in setup)
        try:
            ensure_srs(SRS_PATH, SRS_LOG2_DEGREE, self.log_text)
        except Exception as e:
            log_safe(self.log_text, f"[warn] SRS generation failed or skipped: {e}")

        proofs_dir = Path(base_dir) / PROOFS_DIRNAME
        proofs_dir.mkdir(parents=True, exist_ok=True)
        image_base = os.path.splitext(os.path.basename(self.input_image_path or "input"))[0]
        self.proof_path = str(proofs_dir / f"{image_base}_proof.json")

        self.btn_gen_proof.config(state=tk.DISABLED)
        self.btn_copy_proof.config(state=tk.DISABLED)
        self.btn_save_proof.config(state=tk.DISABLED)

        def _worker():
            try:
                infer_and_build_proof(
                    compiled_model_path=compiled_model_path,
                    settings_path=settings_path,
                    pk_path=pk_path,
                    vk_path=vk_path,
                    input_path=self.input_json_path,
                    output_path=self.proof_path,
                    logger=self.log_text,
                )
            except Exception as e:
                self.after(0, self._prove_done, False, f"Prover failed: {e}")
                return

            try:
                if not self.proof_path or not os.path.exists(self.proof_path):
                    raise FileNotFoundError(f"Proof file not found at: {self.proof_path}")
                if os.path.getsize(self.proof_path) == 0:
                    raise IOError(f"Proof file is empty: {self.proof_path}")
                with open(self.proof_path, "rb") as f:
                    self.proof_bytes = f.read()
                if not self.proof_bytes:
                    raise IOError("Proof file read returned no bytes.")
            except Exception as e:
                self.after(0, self._prove_done, False, f"Could not read proof file: {e}")
                return

            proof_hex = ""
            proof_result = None
            output_hex = None
            try:
                obj = json.loads(self.proof_bytes.decode("utf-8"))
                proof_hex = obj.get("hex_proof") or obj.get("proof") or ""
                ppi = obj.get("pretty_public_inputs", {}) if isinstance(obj, dict) else {}

                def _first_scalar(x):
                    while isinstance(x, list) and x:
                        x = x[0]
                    return x

                proof_result = _first_scalar(ppi.get("rescaled_outputs"))
                out_raw = _first_scalar(ppi.get("outputs"))
                if isinstance(out_raw, str):
                    output_hex = out_raw
            except Exception:
                proof_hex = self.proof_bytes.hex()

            self.proof_hex = proof_hex or ""
            self.proof_result = proof_result
            self.output_hex = output_hex
            self.after(0, self._prove_done, True, None)

        threading.Thread(target=_worker, daemon=True).start()

    def _set_verify_text(self, text):
        self.verify_text.configure(state=tk.NORMAL)
        self.verify_text.delete("1.0", tk.END)
        self.verify_text.insert(tk.END, text)
        self.verify_text.configure(state=tk.DISABLED)

    def _prove_done(self, success, err_msg):
        if success:
            self.proof_text.configure(state=tk.NORMAL)
            self.proof_text.delete("1.0", tk.END)
            self.proof_text.insert(tk.END, self.proof_hex or "")
            self.proof_text.see("1.0")
            self.proof_text.configure(state=tk.DISABLED)

            self.result_var.set(self.proof_result)
            self.btn_copy_proof.config(state=tk.NORMAL)
            self.btn_save_proof.config(state=tk.NORMAL)
            self.btn_gen_proof.config(state=tk.NORMAL)
            self.btn_verify.config(state=tk.NORMAL)

            self.output_hex_text.configure(state=tk.NORMAL)
            self.output_hex_text.delete("1.0", tk.END)
            self.output_hex_text.insert(tk.END, self.output_hex or "—")
            self.output_hex_text.configure(state=tk.DISABLED)
            self.btn_copy_output_hex.config(state=(tk.NORMAL if self.output_hex else tk.DISABLED))
            self._set_verify_text("Ready to verify.")
        else:
            log_safe(self.log_text, f"[ERROR] Proof generation failed: {err_msg}")
            self.btn_gen_proof.config(state=tk.NORMAL)
            messagebox.showerror("Proving failed", f"Failed to create proof:\n{err_msg}")

    def verify_proof(self):
        if not EZKL_AVAILABLE:
            messagebox.showwarning("EZKL missing", "ezkl is not available.")
            return
        if not self.proof_path or not os.path.exists(self.proof_path):
            messagebox.showwarning("No proof", "No proof file found. Generate a proof first.")
            return
        if not (self.settings_path and self.vk_path):
            _, self.settings_path, _, self.vk_path = autodiscover_artifacts(str(self.app_base_dir), self.log_text)

        self.btn_verify.config(state=tk.DISABLED)
        self._set_verify_text("Verifying new proof...")

        def _worker():
            try:
                res = ezkl.verify(self.proof_path, self.settings_path, self.vk_path)
                self.after(0, lambda: (self._set_verify_text(f"Verification result: {res}"),
                                       self.btn_verify.config(state=tk.NORMAL)))
            except Exception as e:
                self.after(0, lambda: (self._set_verify_text(f"Verification failed: {e}"),
                                       self.btn_verify.config(state=tk.NORMAL)))
        threading.Thread(target=_worker, daemon=True).start()

    def copy_proof(self):
        if not self.proof_hex:
            messagebox.showwarning("No proof", "No proof available to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(self.proof_hex)
        self.update()
        messagebox.showinfo("Copied", "Proof hex copied to clipboard.")

    def save_proof(self):
        if not self.proof_bytes:
            messagebox.showwarning("No proof", "No proof available to save.")
            return
        default_name = os.path.basename(self.proof_path) if self.proof_path else "proof.json"
        out_path = filedialog.asksaveasfilename(
            title="Save Proof File",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("Proof files", "*.json"), ("All files", "*.*")]
        )
        if not out_path:
            return
        try:
            with open(out_path, "wb") as f:
                f.write(self.proof_bytes)
            messagebox.showinfo("Saved", f"Saved proof to:\n{os.path.abspath(out_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save proof:\n{e}")

# ---------- entry ----------

if __name__ == "__main__":
    # Run one-time installation (CLI steps) and get the artifact base dir
    try:
        base_dir = run_installation()
    except Exception as e:
        # If install fails early (e.g., missing network.onnx or ezkl CLI), show a dialog then exit
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Setup failed", str(e))
        sys.exit(1)

    app = ImageToTensorApp(app_base_dir=base_dir)
    app.mainloop()
