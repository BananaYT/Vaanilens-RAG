"""
Simple VaaniLens GUI for transcript risk assessment.

Keep this file in the same project folder as assess_transcript.py.
It uses the existing assess_transcript() function and does not modify
your current project files or indexes.
"""

import io
import json
import re
import threading
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
from tkinter import messagebox, ttk


DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_K = 3


def clean_response(response_text: str) -> str:
    """Pretty-print JSON if possible; otherwise show the model response."""
    text = (response_text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.dumps(
                    json.loads(text[start:end + 1]),
                    indent=2,
                    ensure_ascii=False,
                )
            except json.JSONDecodeError:
                pass
        return text or "The model returned an empty response."


class VaaniLensGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VaaniLens — Scam Risk Assessment")
        self.root.geometry("850x700")
        self.root.minsize(650, 550)

        main = ttk.Frame(root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main, text="VaaniLens Risk Assessment",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w")

        ttk.Label(
            main, text="Paste a call transcript below, then click Analyze."
        ).pack(anchor="w", pady=(4, 10))

        self.input_box = tk.Text(main, height=7, wrap="word", font=("Segoe UI", 10))
        self.input_box.pack(fill="x")
        self.input_box.insert(
            "1.0",
            "Your bank account will be blocked today. Tell me your OTP immediately."
        )

        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=10)

        ttk.Label(controls, text="Ollama model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(controls, textvariable=self.model_var, width=20).pack(
            side="left", padx=(6, 14)
        )

        ttk.Label(controls, text="Retrieved items (k):").pack(side="left")
        self.k_var = tk.StringVar(value=str(DEFAULT_K))
        ttk.Entry(controls, textvariable=self.k_var, width=5).pack(
            side="left", padx=(6, 14)
        )

        self.analyze_button = ttk.Button(
            controls, text="Analyze transcript", command=self.start_analysis
        )
        self.analyze_button.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var).pack(
            anchor="w", pady=(0, 6)
        )

        ttk.Label(main, text="VaaniLens risk assessment output:").pack(
            anchor="w", pady=(2, 5)
        )

        output_frame = ttk.Frame(main)
        output_frame.pack(fill="both", expand=True)

        self.output_box = tk.Text(
            output_frame, wrap="word", font=("Consolas", 10),
            state="disabled", padx=10, pady=10
        )
        scrollbar = ttk.Scrollbar(
            output_frame, orient="vertical", command=self.output_box.yview
        )
        self.output_box.configure(yscrollcommand=scrollbar.set)
        self.output_box.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def set_output(self, text):
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", text)
        self.output_box.configure(state="disabled")

    def start_analysis(self):
        transcript = self.input_box.get("1.0", "end").strip()
        model_name = self.model_var.get().strip() or DEFAULT_MODEL

        try:
            k = int(self.k_var.get().strip())
            if k < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid value", "Retrieved items (k) must be a positive integer."
            )
            return

        if not transcript:
            messagebox.showwarning(
                "Transcript required", "Please enter a transcript first."
            )
            return

        self.analyze_button.configure(state="disabled")
        self.status_var.set("Analyzing… this may take a little while.")
        self.set_output("")

        threading.Thread(
            target=self.run_assessment,
            args=(transcript, model_name, k),
            daemon=True,
        ).start()

    def run_assessment(self, transcript, model_name, k):
        # Hide library progress bars and warnings; display only the model result.
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                from assess_transcript import assess_transcript

                response = assess_transcript(
                    transcript=transcript,
                    model_name=model_name,
                    k=k,
                )

            output = clean_response(response)
            self.root.after(0, self.finish_success, output)
        except Exception as exc:
            self.root.after(0, self.finish_error, str(exc))

    def finish_success(self, output):
        self.set_output("=== VaaniLens risk assessment ===\n\n" + output)
        self.status_var.set("Assessment complete")
        self.analyze_button.configure(state="normal")

    def finish_error(self, error):
        self.set_output(
            "Assessment could not be completed.\n\n"
            f"Error: {error}\n\n"
            "Check that Ollama is running, the selected model is installed, "
            "and both VaaniLens FAISS indexes exist."
        )
        self.status_var.set("Assessment failed")
        self.analyze_button.configure(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = VaaniLensGUI(root)
    root.mainloop()
