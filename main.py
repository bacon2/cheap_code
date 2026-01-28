import os
import re
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dotenv import load_dotenv
from openai import OpenAI
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
    r"```(?:\w+)?\s*(.*?)```",
    re.DOTALL
)

# -----------------------------
# UI
# -----------------------------
class CodeProposal(ttk.Frame):
    def __init__(self, parent, code: str, on_replace, on_add):
        super().__init__(parent)
        self.code = code

        # Calculate height: number of lines + 1
        line_count = code.count("\n") + 1
        
        self.text = tk.Text(
            self,
            height=line_count,
            wrap="none",
            font=("Courier New", 10),
            padx=5,
            pady=5,
            highlightthickness=1 # Border for code blocks
        )
        self.text.insert("1.0", code)
        self.text.configure(state="disabled")
        self.text.pack(fill="x", padx=5, pady=(5, 2))

        button_row = ttk.Frame(self)
        button_row.pack(anchor="e", padx=5, pady=(0, 5))

        ttk.Button(button_row, text="Replace", command=lambda: on_replace(code)).pack(side="right", padx=2)
        ttk.Button(button_row, text="Add", command=lambda: on_add(code)).pack(side="right", padx=2)

class ChatUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chat + Code UI")
        self.root.geometry("1100x650")

        self.conversation = [
            {"role": "system", "content": "You are a helpful coding assistant. Your code snippets must never contain ellipses, or any comment meant to represent code. "}
        ]

        self.token_queue = queue.Queue()
        self.is_generating = False
        self.current_assistant_buffer = ""

        self.code_contents = ""

        # ---- persistence helpers ----
        self.auto_save_path: str | None = None
        self.auto_save_after_id: str | None = None
        self.auto_save_interval_ms = 30_000
        
        # Default models
        self.chat_model = tk.StringVar(value="llama-3.3-70b-versatile")
        self.helper_model = tk.StringVar(value="llama-3.3-70b-versatile")

        self._build_layout()
        self._bind_keys()
        self._poll_tokens()

    # -----------------------------
    # Layout
    # -----------------------------
    def _select_all_code(self, event=None):
        self.code_panel.tag_add("sel", "1.0", "end")

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

        # Scrollable canvas
        self.chat_canvas = tk.Canvas(chat_frame)
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_canvas.yview)
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")

        # Inner frame to hold all messages
        self.chat_inner_frame = ttk.Frame(self.chat_canvas)
        self.chat_canvas.create_window((0, 0), window=self.chat_inner_frame, anchor="nw")

        # Update scroll region when content changes
        self.chat_inner_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )

        # Track the current message label for streaming
        self.current_message_label = None
        
        # Get background color from a standard widget
        self.bg_color = self.root.cget("background")

        # ----------------------
        # Code Panel
        # ----------------------
        code_frame = ttk.Frame(main_pane)
        code_frame.rowconfigure(0, weight=1)
        code_frame.columnconfigure(0, weight=1)

        self.code_panel = tk.Text(code_frame, wrap="none")
        self.code_panel.grid(row=0, column=0, sticky="nsew")
        
        # Bind to sync code_contents when user edits
        self.code_panel.bind("<<Modified>>", self._on_code_modified)

        # Add this line to bind Ctrl+A to select all code
        self.code_panel.bind_all("<Control-a>", self._select_all_code)
        
        

        code_scroll = ttk.Scrollbar(code_frame, command=self.code_panel.yview)
        code_scroll.grid(row=0, column=1, sticky="ns")
        self.code_panel["yscrollcommand"] = code_scroll.set

        main_pane.add(chat_frame)
        main_pane.add(code_frame)

        # ----------------------
        # Input Box
        # ----------------------
        self.input_box = tk.Text(self.root, height=5, wrap="word")
        self.input_box.grid(row=2, column=0, sticky="nsew")

        self.root.update_idletasks()
        main_pane.sashpos(0, int(self.root.winfo_width() * 0.70))
    
    def _get_client_for_model(self, model: str):
        """Return the appropriate client for the given model."""
        if model in OPENAI_MODELS:
            return openai_client
        else:
            return groq_client
    
    def _get_max_tokens_for_model(self, model: str):
        """Return appropriate max_tokens for the given model."""
        if model in OPENAI_MODELS:
            return 16000
        else:
            return 8000
    
    def _uses_max_completion_tokens(self, model: str):
        """Check if model uses max_completion_tokens instead of max_tokens."""
        return model in ["gpt-5.2", "gpt-5-mini"]

    def _bind_keys(self):
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)
    
    def _on_code_modified(self, event):
        """Sync code_contents when user edits the code panel."""
        if self.code_panel.edit_modified():
            self.code_contents = self.code_panel.get("1.0", "end-1c")
            self.code_panel.edit_modified(False)

    # -----------------------------
    # Input
    # -----------------------------

    def _on_enter(self, event):
        if self.is_generating:
            return "break"

        text = self.input_box.get("1.0", "end-1c").strip()
        if not text:
            return "break"

        self.input_box.delete("1.0", "end")
        self._append_chat("You", text)
        
        # Add user message to conversation
        self.conversation.append({"role": "user", "content": text})
        
        # Add invisible context about current code if it exists
        if self.code_contents.strip():
            self.conversation.append({
                "role": "user",
                "content": f"Here's my current code:\n{self.code_contents}"
            })

        self._start_generation()
        return "break"

    # -----------------------------
    # Chat Generation
    # -----------------------------

    def _start_generation(self):
        self.is_generating = True
        self.input_box.configure(state="disabled")
        self.current_assistant_buffer = ""

        threading.Thread(
            target=self._stream_chat_response, daemon=True
        ).start()

    def _stream_chat_response(self):
        try:
            model = self.chat_model.get()
            client = self._get_client_for_model(model)
            max_tokens = self._get_max_tokens_for_model(model)
            
            # Build request parameters
            params = {
                "model": model,
                "messages": self.conversation,
                "stream": True,
            }
            
            # Use max_completion_tokens for GPT-5.2 and GPT-5-mini
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

        # Replace the streaming label with parsed content
        self._replace_with_parsed_message(assistant_text)

    def _replace_with_parsed_message(self, text: str):
        """Parse the message and replace code blocks with interactive widgets."""
        if self.current_message_label:
            self.current_message_label.destroy()
            self.current_message_label = None

        # Create a container for this message
        message_container = ttk.Frame(self.chat_inner_frame)
        message_container.pack(fill="x", padx=5, pady=5)

        # Add sender as copyable text
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

        # Split text by code blocks
        parts = self._split_text_and_code(text)
        
        for part_type, content in parts:
            if part_type == "text":
                if content.strip():
                    text_widget = tk.Text(
                        message_container,
                        height=content.count("\n") + 1 + int(len(content) * 0.012),
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

        # Scroll to bottom
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def _split_text_and_code(self, text: str):
        """Split text into alternating text and code blocks."""
        parts = []
        last_end = 0
        
        for match in CODE_BLOCK_REGEX.finditer(text):
            # Add text before code block
            if match.start() > last_end:
                parts.append(("text", text[last_end:match.start()]))
            
            # Add code block
            parts.append(("code", match.group(1)))
            last_end = match.end()
        
        # Add remaining text after last code block
        if last_end < len(text):
            parts.append(("text", text[last_end:]))
        
        return parts

    # -----------------------------
    # Code Handling (Line-Number Based Approach)
    # -----------------------------

    def _add_line_numbers(self, code: str) -> str:
        """Prepend line numbers to each line of code."""
        lines = code.split("\n")
        numbered_lines = [f"{i+1} | {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def _add_code_via_helper(self, new_code: str):
        """Ask helper model for insertion point and lines to delete, then apply."""
        if not self.code_contents.strip():
            self._replace_code_panel(new_code)
            return

        # Get insertion instructions from helper model
        insertion_data = self._get_insertion_instructions(new_code)
        
        if not insertion_data:
            self._append_chat("System", "Helper model did not return valid instructions.")
            return

        try:
            # Apply the edit first
            updated_code = self._apply_line_based_edit(
                insertion_data["insertion_point"],
                insertion_data["delete_lines"],
                new_code,
                insertion_data["indent_spaces"]
            )

            # Deduplicate lines after applying edits
            updated_code = self._dedupe_post_edit(
                updated_code,
                insertion_data["insertion_point"],
                new_code
            )

            self.code_contents = updated_code
            self.code_panel.delete("1.0", "end")
            self.code_panel.insert("1.0", updated_code)
            self._append_chat("System", "Code updated successfully!")

        except Exception as e:
            self._append_chat("System", f"Edit failed: {e}")


    def _get_insertion_instructions(self, new_code: str) -> dict | None:
        numbered_code = self._add_line_numbers(self.code_contents)

        system_prompt = (
            "You are a deterministic code editing assistant.\n"
            "You must respond ONLY with JSON in the exact format specified.\n\n"
            "JSON format:\n"
            "{\n"
            '  "anchor_line": <line_number>,\n'
            '  "replace_count": <number_of_lines>,\n'
            '  "indent_spaces": <number_of_spaces>\n'
            "}\n\n"
            "Rules:\n"
            "- anchor_line must be a line number from the ORIGINAL code\n"
            "- replace_count is how many lines AFTER anchor_line to replace\n"
            "- Use replace_count = 0 for pure insertion\n"
            "- indent_spaces must match surrounding indentation\n"
            "- Do NOT list delete lines\n"
            "- No explanations. JSON only."
        )

        user_prompt = (
            "ORIGINAL CODE (with line numbers):\n"
            f"{numbered_code}\n\n"
            "NEW CODE TO INSERT OR REPLACE WITH:\n"
            f"{new_code}\n"
        )

        model = self.helper_model.get()
        client = self._get_client_for_model(model)
        max_tokens = self._get_max_tokens_for_model(model)

        print("\n===== HELPER INITIAL PROMPT =====")
        print(system_prompt)
        print(user_prompt)

        raw = self._call_helper_and_parse(
            client,
            model,
            max_tokens,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        print("Initial helper JSON (raw):", raw)

        if not raw:
            print("❌ No valid JSON returned")
            return None

        try:
            anchor_line = raw["anchor_line"]
            replace_count = raw["replace_count"]
            indent_spaces = raw["indent_spaces"]
        except KeyError as e:
            print("❌ Missing required key:", e)
            return None

        if (
            not isinstance(anchor_line, int)
            or anchor_line < 0
            or not isinstance(replace_count, int)
            or replace_count < 0
            or not isinstance(indent_spaces, int)
            or indent_spaces < 0
        ):
            print("❌ Invalid values in helper JSON:", raw)
            return None

        delete_lines = (
            list(range(anchor_line + 1, anchor_line + 1 + replace_count))
            if replace_count > 0
            else []
        )

        result = {
            "insertion_point": anchor_line,
            "delete_lines": delete_lines,
            "indent_spaces": indent_spaces,
        }

        print("✅ Final derived insertion instructions:", result)
        return result




    def _get_modification_context(self, updated_code: str, insertion_point: int, 
                                   delete_lines: list, new_code: str, indent_spaces: int) -> str:
        """
        Extract a snippet showing the modification with lines before and after context.
        Returns numbered lines.
        """
        lines = updated_code.split("\n")
        
        # Calculate where the new code actually is in the updated file
        # We need to figure out the range of lines that were affected
        
        # Start by finding where we inserted
        num_deleted = len(delete_lines)
        num_inserted = len(new_code.split("\n"))
        
        # The insertion point in the original file
        original_insertion = insertion_point
        
        # In the new file, after deleting lines before the insertion point
        deleted_before = sum(1 for line_num in delete_lines if line_num <= insertion_point)
        adjusted_insertion = insertion_point - deleted_before
        
        # The new code spans from adjusted_insertion to adjusted_insertion + num_inserted
        start_of_new_code = adjusted_insertion
        end_of_new_code = adjusted_insertion + num_inserted
        
        # Get lines before and after
        context_start = max(0, start_of_new_code - num_inserted * 4)
        context_end = min(len(lines), end_of_new_code + num_inserted * 4)
        
        # Extract the context
        context_lines = []
        if context_start > 0:
            context_lines += "... (earlier code)"
        context_lines += lines[context_start:context_end]
        if context_end < len(lines):
            context_lines += "... (later code)"
        # Add line numbers (starting from context_start + 1)
        numbered_context = []
        for i, line in enumerate(context_lines):
            line_num = context_start + i + 1
            numbered_context.append(f"{line_num} | {line}")
        
        return "\n".join(numbered_context)

    def _call_helper_and_parse(self, client, model: str, max_tokens: int, conversation: list) -> dict | None:
        """Call the helper model and parse JSON response."""
        params = {
            "model": model,
            "messages": conversation,
        }
        
        if self._uses_max_completion_tokens(model):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        
        try:
            response = client.chat.completions.create(**params)
            response_text = response.choices[0].message.content or ""
            
            print("Helper response:", response_text)
            
            raw = self._extract_json_from_text(response_text)
            return raw

            
        except Exception as e:
            print(f"Error calling helper: {e}")
            return None
    

    def _extract_json_from_text(self, text: str) -> dict | None:
        """
        Extract the first valid JSON object from text.
        Does NOT validate schema.
        """
        import json

        if not text:
            return None

        # Fast path: whole string is JSON
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass

        # Fallback: find first {...} block
        start = text.find("{")
        while start != -1:
            end = text.find("}", start)
            while end != -1:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    end = text.find("}", end + 1)
            start = text.find("{", start + 1)

        return None

    
    def _validate_json_format(self, data: dict) -> bool:
        """Validate that JSON has required fields with correct types."""
        if "insertion_point" not in data or "delete_lines" not in data or "indent_spaces" not in data:
            return False
        
        if not isinstance(data["insertion_point"], int):
            return False
        
        if not isinstance(data["delete_lines"], list):
            return False
        
        if not isinstance(data["indent_spaces"], int):
            return False
        
        return True
    
    def _dedupe_post_edit(self, updated_code: str, insertion_point: int, new_code: str) -> str:
        """
        Remove a single duplicate line if the first non-comment line of new_code
        is identical to the first non-comment line immediately before insertion_point.
        """
        lines = updated_code.splitlines()
        new_lines = [l for l in new_code.splitlines() if l.strip() and not l.lstrip().startswith("#")]
        
        if not new_lines:
            return updated_code  # nothing to compare
        
        # Find the first non-comment line above insertion_point
        idx = insertion_point - 1  # convert to 0-based
        while idx >= 0 and (lines[idx].strip() == "" or lines[idx].lstrip().startswith("#")):
            idx -= 1
        
        if idx >= 0:
            existing_line = lines[idx].strip()
            first_new_line = new_lines[0].strip()
            # Also allow deduping for function/class headers even if whitespace differs
            both_headers = first_new_line.startswith(("def ", "class ")) and existing_line.startswith(("def ", "class "))
            if existing_line == first_new_line or both_headers:
                del lines[idx]
        
        return "\n".join(lines)




    def _apply_line_based_edit(self, insertion_point: int, delete_lines: list, new_code: str, indent_spaces: int) -> str:
        """Apply insertion and deletion based on line numbers, with indentation."""
        lines = self.code_contents.split("\n")
        
        # Validate line numbers
        if insertion_point < 0 or insertion_point > len(lines):
            raise ValueError(f"Invalid insertion point: {insertion_point}")
        
        for line_num in delete_lines:
            if line_num < 1 or line_num > len(lines):
                raise ValueError(f"Invalid line number to delete: {line_num}")
        
        # Convert delete_lines to 0-indexed and sort in reverse order
        delete_indices = sorted([line_num - 1 for line_num in delete_lines], reverse=True)
        
        # Adjust insertion point if we're deleting lines before it
        adjusted_insertion = insertion_point
        for idx in delete_indices:
            if idx < insertion_point:
                adjusted_insertion -= 1
        
        # Delete lines (in reverse order to preserve indices)
        for idx in delete_indices:
            del lines[idx]
        
        # Apply indentation to new code
        indent = " " * indent_spaces
        new_lines = new_code.split("\n")
        indented_lines = [indent + line for line in new_lines]
        
        # Insert new code at the adjusted position
        lines[adjusted_insertion:adjusted_insertion] = indented_lines
        
        return "\n".join(lines)

    def _replace_code_panel(self, code: str):
        self.code_contents = code
        self.code_panel.delete("1.0", "end")
        self.code_panel.insert("1.0", code)

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
                raw = f.read().strip()
            if not raw:
                raise ValueError("File is empty.")
            loaded = json.loads(raw)
        except json.JSONDecodeError as e:
            messagebox.showerror("Load error", f"Invalid JSON:\n{e}")
            return
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
    
        if not isinstance(loaded, list):
            messagebox.showerror(
                "Load error",
                f"File must contain a list of messages, got {type(loaded).__name__}."
            )
            return
    
        self.conversation = loaded
        self.auto_save_path = path
        self._redraw_chat()
        messagebox.showinfo("Loaded", f"Conversation loaded from\n{path}")
    
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

    # -----------------------------
    # UI Helpers
    # -----------------------------

    def _append_chat(self, sender: str, text: str):
        """Modified for tighter user message spacing."""
        container = ttk.Frame(self.chat_inner_frame)
        container.pack(fill="x", padx=10, pady=5)

        header = tk.Text(container, height=1, font=("TkDefaultFont", 10, "bold"),
                        relief="flat", background=self.bg_color, highlightthickness=0)
        header.insert("1.0", f"{sender}:")
        header.configure(state="disabled")
        header.pack(fill="x")

        # Dynamic height for content
        line_count = text.count("\n") + 1
        body = tk.Text(container, height=line_count, wrap="word", relief="flat",
                      background=self.bg_color, highlightthickness=0)
        body.insert("1.0", text)
        body.configure(state="disabled")
        body.pack(fill="x", padx=5)
        
        if sender == "Assistant":
            self.current_message_label = body
            self.current_message_container = container

        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def _append_to_last_chat(self, text: str):
        """Append text and grow the text widget height dynamically."""
        if self.current_message_label:
            self.current_message_label.configure(state="normal")
            self.current_message_label.insert("end", text)
            
            line_count = int(self.current_message_label.index("end-1c").split(".")[0])
            self.current_message_label.configure(height=line_count, state="disabled")
            
            self.chat_canvas.update_idletasks()
            self.chat_canvas.yview_moveto(1.0)


if __name__ == "__main__":
    root = tk.Tk()
    ChatUI(root)
    root.mainloop()

