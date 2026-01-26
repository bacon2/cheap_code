import os
import re
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dotenv import load_dotenv
from openai import OpenAI
from unidiff import PatchSet
import difflib
import json

# -----------------------------
# Environment / Client Setup
# -----------------------------

load_dotenv()

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Model configurations
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-4.1",
]

ALL_MODELS = GROQ_MODELS + OPENAI_MODELS

CODE_BLOCK_REGEX = re.compile(
    r"```(?:\w+)?\n(.*?)```",
    re.DOTALL
)

# -----------------------------
# UI
# -----------------------------
class CodeProposal(ttk.Frame):
    def __init__(self, parent, code: str, on_replace, on_add):
        super().__init__(parent)

        self.code = code

        self.text = tk.Text(
            self,
            height=min(12, code.count("\n") + 2),
            wrap="none",
            font=("Courier New", 10)
        )
        self.text.insert("1.0", code)
        self.text.configure(state="disabled")
        self.text.pack(fill="x", padx=5, pady=5)

        button_row = ttk.Frame(self)
        button_row.pack(anchor="e", padx=5, pady=(0, 5))

        ttk.Button(
            button_row, text="Replace",
            command=lambda: on_replace(code)
        ).pack(side="right", padx=2)

        ttk.Button(
            button_row, text="Add",
            command=lambda: on_add(code)
        ).pack(side="right", padx=2)

class ChatWithCode(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Scrollable canvas
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Inner frame that holds all messages
        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # Resize scrollregion when inner_frame changes
        self.inner_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def add_message_with_code(self, text: str, code: str, on_replace, on_add):
        # Text explanation
        text_label = tk.Label(
            self.inner_frame,
            text=text,
            wraplength=600,
            justify="left"
        )
        text_label.pack(fill="x", padx=5, pady=(5, 0))

        # CodeProposal widget
        proposal = CodeProposal(self.inner_frame, code, on_replace, on_add)
        proposal.pack(fill="x", padx=5, pady=(0, 10))

        # Scroll to bottom
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)


# ------------------------------------------------------------------
#  ChatUI  –  conversation-only persistence
# ------------------------------------------------------------------
class ChatUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chat + Code UI")
        self.root.geometry("1100x650")

        self.conversation = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        self.token_queue = queue.Queue()
        self.is_generating = False
        self.current_assistant_buffer = ""

        self.code_contents = ""

        # ---- persistence helpers ----
        self.auto_save_path: str | None = None   # last saved file
        self.auto_save_after_id: str | None = None
        self.auto_save_interval_ms = 30_000      # 30-second auto-save

        # Default models
        self.chat_model = tk.StringVar(value="llama-3.3-70b-versatile")
        self.helper_model = tk.StringVar(value="llama-3.3-70b-versatile")

        self._build_layout()
        self._bind_keys()
        self._poll_tokens()

        # kick off auto-save loop
        self._schedule_auto_save()

    # -----------------------------
    # Layout (unchanged except for menu)
    # -----------------------------
    def _build_layout(self) -> None:
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)

        # ----------------------
        # Model Selection Bar
        # ----------------------
        model_frame = ttk.Frame(self.root)
        model_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Label(model_frame, text="Chat Model:").pack(side="left", padx=(0, 5))
        chat_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.chat_model,
            values=ALL_MODELS,
            state="readonly",
            width=40
        )
        chat_dropdown.pack(side="left", padx=(0, 20))

        ttk.Label(model_frame, text="Helper Model:").pack(side="left", padx=(0, 5))
        helper_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.helper_model,
            values=ALL_MODELS,
            state="readonly",
            width=40
        )
        helper_dropdown.pack(side="left")

        # ----------------------
        # Menu bar
        # ----------------------
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save conversation…", command=self.save_conversation)
        file_menu.add_command(label="Load conversation…", command=self.load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Export markdown…", command=self.export_markdown)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.grid(row=1, column=0, sticky="nsew")

        # ----------------------
        # Chat + Code Proposals
        # ----------------------
        chat_frame = ttk.Frame(main_pane)
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)

        self.chat_canvas = tk.Canvas(chat_frame)
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_canvas.yview)
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")

        self.chat_inner_frame = ttk.Frame(self.chat_canvas)
        self.chat_canvas.create_window((0, 0), window=self.chat_inner_frame, anchor="nw")
        self.chat_inner_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        self.bg_color = self.root.cget("background")

        # ----------------------
        # Code Panel
        # ----------------------
        code_frame = ttk.Frame(main_pane)
        code_frame.rowconfigure(0, weight=1)
        code_frame.columnconfigure(0, weight=1)

        self.code_panel = tk.Text(code_frame, wrap="none")
        self.code_panel.grid(row=0, column=0, sticky="nsew")
        self.code_panel.bind("<<Modified>>", self._on_code_modified)

        code_scroll = ttk.Scrollbar(code_frame, command=self.code_panel.yview)
        code_scroll.grid(row=0, column=1, sticky="ns")
        self.code_panel["yscrollcommand"] = code_scroll.set

        main_pane.add(chat_frame, weight=1)
        main_pane.add(code_frame, weight=1)

        # ----------------------
        # Input Box
        # ----------------------
        self.input_box = tk.Text(self.root, height=5, wrap="word")
        self.input_box.grid(row=2, column=0, sticky="nsew")

    # -----------------------------
    # Model helpers (unchanged)
    # -----------------------------
    def _get_client_for_model(self, model: str):
        return openai_client if model in OPENAI_MODELS else groq_client

    def _get_max_tokens_for_model(self, model: str):
        return 16000 if model in OPENAI_MODELS else 8000

    def _uses_max_completion_tokens(self, model: str):
        return model in ["gpt-5.2", "gpt-5-mini"]

    # -----------------------------
    # Input / generation (unchanged)
    # -----------------------------
    def _bind_keys(self):
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)

    def _on_code_modified(self, event):
        if self.code_panel.edit_modified():
            self.code_contents = self.code_panel.get("1.0", "end-1c")
            self.code_panel.edit_modified(False)

    def _on_enter(self, event):
        if self.is_generating:
            return "break"
        text = self.input_box.get("1.0", "end-1c").strip()
        if not text:
            return "break"
        self.input_box.delete("1.0", "end")
        self._append_chat("You", text)
        self.conversation.append({"role": "user", "content": text})
        if self.code_contents.strip():
            self.conversation.append({
                "role": "user",
                "content": f"Here's my current code:\n{self.code_contents}"
            })
        self._start_generation()
        return "break"

    def _start_generation(self):
        self.is_generating = True
        self.input_box.configure(state="disabled")
        self.current_assistant_buffer = ""
        self._append_chat("Assistant", "")
        threading.Thread(target=self._stream_chat_response, daemon=True).start()

    def _stream_chat_response(self):
        try:
            model = self.chat_model.get()
            client = self._get_client_for_model(model)
            max_tokens = self._get_max_tokens_for_model(model)
            params = {
                "model": model,
                "messages": self.conversation,
                "stream": True,
            }
            if self._uses_max_completion_tokens(model):
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
            stream = client.chat.completions.create(**params)
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    self.token_queue.put(delta)
        except Exception as e:
            self.token_queue.put(f"\n[Error: {e}]")
        finally:
            self.token_queue.put(None)

    def _poll_tokens(self):
        try:
            while True:
                token = self.token_queue.get_nowait()
                if token is None:
                    self._finalize_assistant_message()
                    break
                self.current_assistant_buffer += token
                self._append_to_last_chat(token)
        except queue.Empty:
            pass
        self.root.after(20, self._poll_tokens)

    def _finalize_assistant_message(self):
        assistant_text = self.current_assistant_buffer
        self.conversation.append({"role": "assistant", "content": assistant_text})
        self.is_generating = False
        self.input_box.configure(state="normal")
        self._replace_with_parsed_message(assistant_text)

    # -----------------------------
    # Parsing / UI helpers (unchanged)
    # -----------------------------
    def _replace_with_parsed_message(self, text: str):
        if hasattr(self, "current_message_container") and self.current_message_container:
            self.current_message_container.destroy()
            self.current_message_container = None
            self.current_message_label = None


        message_container = ttk.Frame(self.chat_inner_frame)
        message_container.pack(fill="x", padx=5, pady=5)

        # ---- Sender line (always exactly 1 line) ----
        sender_text = tk.Text(
            message_container,
            height=1,
            wrap="none",
            font=("TkDefaultFont", 10, "bold"),
            relief="flat",
            background=self.bg_color
        )
        sender_text.insert("1.0", "Assistant:")
        sender_text.configure(state="disabled")
        sender_text.pack(fill="x")

        # ---- Content ----
        parts = self._split_text_and_code(text)

        for part_type, content in parts:
            if part_type == "text" and content.strip():
                text_widget = tk.Text(
                    message_container,
                    wrap="word",
                    relief="flat",
                    background=self.bg_color
                )
                text_widget.insert("1.0", content)
                text_widget.configure(state="disabled")
                text_widget.pack(fill="x", padx=5, pady=2)

            elif part_type == "code":
                proposal = CodeProposal(
                    parent=message_container,
                    code=content,
                    on_replace=self._replace_code_panel,
                    on_add=self._add_code_via_helper,
                )
                proposal.pack(fill="x", pady=5)

        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def _split_text_and_code(self, text: str):
        parts = []
        last_end = 0
        for match in CODE_BLOCK_REGEX.finditer(text):
            if match.start() > last_end:
                parts.append(("text", text[last_end:match.start()]))
            parts.append(("code", match.group(1)))
            last_end = match.end()
        if last_end < len(text):
            parts.append(("text", text[last_end:]))
        return parts

    # -----------------------------
    # Code handling (unchanged)
    # -----------------------------
    def _extract_code_blocks(self, text: str) -> list[str]:
        return CODE_BLOCK_REGEX.findall(text)

    def _add_code_via_helper(self, new_code: str):
        if not self.code_contents.strip():
            self._replace_code_panel(new_code)
            return
        diff = self._generate_diff_with_helper(old_code=self.code_contents, new_code=new_code)
        if not diff.strip():
            return
        try:
            patched = self._apply_unified_diff(self.code_contents, diff)
        except Exception as e:
            self._append_chat("System", diff)
            self._append_chat("System", f"Diff failed:\n{e}")
            return
        self.code_contents = patched
        self.code_panel.delete("1.0", "end")
        self.code_panel.insert("1.0", patched)

    def _replace_code_panel(self, code: str):
        self.code_contents = code
        self.code_panel.delete("1.0", "end")
        self.code_panel.insert("1.0", code)

    def _generate_diff_with_helper(self, old_code: str, new_code: str) -> str:
        system_prompt = (
            "You generate unified diffs that will be applied automatically.\n"
            "The diff must apply cleanly.\n"
            "Output ONLY a unified diff.\n"
            "Do NOT include explanations, prose, or markdown.\n"
            "Always include file headers using 'a/code.py' and 'b/code.py'.\n"
            "If no changes are required, output an empty response."
        )
        user_prompt = (
            "CURRENT FILE:\n"
            f"{old_code}\n\n"
            "NEW CODE TO INTEGRATE:\n"
            f"{new_code}\n"
        )
        model = self.helper_model.get()
        client = self._get_client_for_model(model)
        max_tokens = self._get_max_tokens_for_model(model)
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._uses_max_completion_tokens(model):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        response = client.chat.completions.create(**params)
        diff_text = response.choices[0].message.content or ""
        return diff_text.strip()

    def _fix_diff_hunk_counts(self, diff_text: str) -> str:
        lines = diff_text.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('@@'):
                match = re.match(r'@@ -(\d+),?\d* \+(\d+),?\d* @@', line)
                if match:
                    old_start = match.group(1)
                    new_start = match.group(2)
                    hunk_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith('@@'):
                        hunk_lines.append(lines[j])
                        j += 1
                    old_count = sum(1 for l in hunk_lines if l.startswith('-') or l.startswith(' '))
                    new_count = sum(1 for l in hunk_lines if l.startswith('+') or l.startswith(' '))
                    fixed_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"
                    fixed_lines.append(fixed_header)
                    fixed_lines.extend(hunk_lines)
                    i = j
                else:
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
        return '\n'.join(fixed_lines)

    def _apply_unified_diff(self, old_text: str, diff_text: str) -> str:
        fixed_diff = self._fix_diff_hunk_counts(diff_text)
        return self._apply_patch_to_text(old_text, fixed_diff)

    def _apply_patch_to_text(self, old_text: str, diff_text: str) -> str:
        old_lines = old_text.splitlines(keepends=True)
        patch = PatchSet(diff_text)
        if len(patch) != 1:
            raise ValueError("Patch must have exactly one file for code panel")
        patched_file = patch[0]
        new_lines = []
        old_idx = 0
        for hunk in patched_file:
            while old_idx < hunk.source_start - 1:
                new_lines.append(old_lines[old_idx])
                old_idx += 1
            for line in hunk:
                if line.is_added:
                    new_lines.append(line.value)
                elif line.is_context:
                    new_lines.append(old_lines[old_idx])
                    old_idx += 1
                elif line.is_removed:
                    old_idx += 1
        new_lines.extend(old_lines[old_idx:])
        return "".join(new_lines)

    # -----------------------------
    # Persistence
    # -----------------------------
    def save_conversation(self, path: str | None = None):
        if path is None:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not path:
                return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.conversation, f, indent=2, ensure_ascii=False)
            self.auto_save_path = path
            messagebox.showinfo("Saved", f"Conversation saved to\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def load_conversation(self):
        path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("File must contain a list of messages.")
            self.conversation = loaded
            self.auto_save_path = path
            self._redraw_chat()
            messagebox.showinfo("Loaded", f"Conversation loaded from\n{path}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def export_markdown(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            lines = []
            for msg in self.conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    continue
                name = "You" if role == "user" else "Assistant"
                lines.append(f"**{name}:**\n{content}\n")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Exported", f"Markdown exported to\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _redraw_chat(self):
        for w in self.chat_inner_frame.winfo_children():
            w.destroy()
        self.current_message_label = None
        for msg in self.conversation:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                continue
            sender = "You" if role == "user" else "Assistant"
            if role == "assistant":
                self._replace_with_parsed_message(content)
            else:
                self._append_chat(sender, content)

    def _schedule_auto_save(self):
        if self.auto_save_path and self.conversation:
            try:
                with open(self.auto_save_path, "w", encoding="utf-8") as f:
                    json.dump(self.conversation, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        self.auto_save_after_id = self.root.after(self.auto_save_interval_ms, self._schedule_auto_save)

    # -----------------------------
    # UI Helpers (unchanged)
    # -----------------------------
    def _append_chat(self, sender: str, text: str):
        message_container = ttk.Frame(self.chat_inner_frame)
        message_container.pack(fill="x", padx=5, pady=5)

        self.current_message_container = message_container


        # ---- Sender line (fixed, 1 row) ----
        sender_text = tk.Text(
            message_container,
            height=1,
            wrap="none",
            font=("TkDefaultFont", 10, "bold"),
            relief="flat",
            background=self.bg_color
        )
        sender_text.insert("1.0", f"{sender}:")
        sender_text.configure(state="disabled")
        sender_text.pack(fill="x")

        # ---- Message body (streaming target) ----
        body_text = tk.Text(
            message_container,
            wrap="word",
            relief="flat",
            background=self.bg_color
        )

        self.current_message_label = body_text

        body_text.insert("1.0", text)
        body_text.configure(state="disabled")
        body_text.pack(fill="x", padx=5, pady=(2, 0))

        # IMPORTANT:
        # current_message_label must point to the *body* text widget
        self.current_message_label = body_text

        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)


    def _append_to_last_chat(self, text: str):
        if self.current_message_label and self.is_generating:
            self.current_message_label.configure(state="normal")
            self.current_message_label.insert("end", text)
            self.current_message_label.configure(state="disabled")
            self.chat_canvas.update_idletasks()
            self.chat_canvas.yview_moveto(1.0)




if __name__ == "__main__":
    root = tk.Tk()
    ChatUI(root)
    root.mainloop()