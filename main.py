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
            {"role": "system", "content": """
You are a coding assistant inside a UI that extracts and applies code blocks verbatim. The user may copy/paste code blocks directly into files. Therefore, code blocks must be **complete, explicit, and directly runnable** (or clearly minimal but still syntactically valid).  

**Critical rule: Never use placeholder elisions** in code blocks or patch instructions. This includes (but is not limited to):  
- `...` or `…`  
- `# ...` / `// ...` / `/* ... */`  
- “(other code)”, “existing code”, “rest of file”, “unchanged”, “omitted”, “same as above/below”  
- “insert here”, “fill in”, “TODO: add …” (unless the user explicitly asked for TODOs)

If a full-file rewrite would be too long, do **one** of these instead:
1) Output **multiple complete code blocks** representing each full function/class/module that must be added or replaced, and explicitly name what each block replaces; or  
2) Ask a **single clarifying question** to reduce scope so you can provide complete code.

When you provide code edits:
- Ensure the code in each fenced block is self-contained and contains all required imports/definitions for that block to work as presented.  
- If you reference “add this to X”, you must show the exact insertion target by function/class name (and if needed, quote the exact original lines you’re replacing) — but do not use ellipses.

Formatting rules:
- Critical rule: Never use placeholder comments to represent code that has already been written
- Use fenced code blocks only for real code or real patches.
- If you cannot comply with the no-elision rule, stop and ask for the missing file/context instead of guessing.
             """}
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
        self.code_panel.bind("<Control-a>", lambda e: (self.code_panel.tag_add("sel", "1.0", "end"), "break")[1])
        
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

        # Try to apply the insertion and deletion
        try:
            updated_code = self._apply_line_based_edit(
                insertion_data["insertion_point"],
                insertion_data["delete_lines"],
                new_code,
                insertion_data["indent_spaces"]
            )
            self.code_contents = updated_code
            self.code_panel.delete("1.0", "end")
            self.code_panel.insert("1.0", updated_code)
            self._append_chat("System", "Code updated successfully!")
        except Exception as e:
            self._append_chat("System", f"Edit failed: {e}")

    def _get_insertion_instructions(self, new_code: str) -> dict | None:
        """Ask helper model for insertion point and lines to delete using line numbers, with verification."""
        # Add line numbers to current code
        numbered_code = self._add_line_numbers(self.code_contents)
        
        system_prompt = (
            "You are a code editing assistant. Your job is to determine where to insert new code "
            "and which lines (if any) to delete from the existing code.\n\n"
            "You will receive:\n"
            "1. The current code with line numbers (format: '1 | code here')\n"
            "2. New code to insert\n\n"
            "You must respond ONLY with JSON in this exact format:\n"
            "{\n"
            '  "insertion_point": <line_number>,\n'
            '  "delete_lines": [<line_numbers>],\n'
            '  "indent_spaces": <number_of_spaces>\n'
            "}\n\n"
            "Rules:\n"
            "- insertion_point: The line number AFTER which to insert the new code (use 0 to insert at the beginning)\n"
            "- delete_lines: Array of line numbers to delete (can be empty [] if just inserting)\n"
            "- indent_spaces: Number of spaces to add to the beginning of each line of inserted code (to match surrounding indentation)\n"
            "- Use the CURRENT line numbers you see - don't worry about how insertion affects numbering\n"
            "- If replacing code, include those line numbers in delete_lines\n"
            "- If just inserting, leave delete_lines empty\n"
            "- Look at the indentation of surrounding code to determine indent_spaces\n\n"
            "Example 1 - Replace lines 5-7 with 8 spaces of indentation:\n"
            '{"insertion_point": 4, "delete_lines": [5, 6, 7], "indent_spaces": 8}\n\n'
            "Example 2 - Insert after line 10 with 4 spaces:\n"
            '{"insertion_point": 10, "delete_lines": [], "indent_spaces": 4}\n\n'
            "Example 3 - Insert at beginning with no indentation:\n"
            '{"insertion_point": 0, "delete_lines": [], "indent_spaces": 0}\n\n'
            "Respond with ONLY the JSON, no explanations."
        )

        user_prompt = (
            "CURRENT CODE (with line numbers):\n"
            f"{numbered_code}\n\n"
            "NEW CODE TO INSERT:\n"
            f"{new_code}\n"
        )

        model = self.helper_model.get()
        client = self._get_client_for_model(model)
        max_tokens = self._get_max_tokens_for_model(model)
        
        # First attempt
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        data = self._call_helper_and_parse(client, model, max_tokens, conversation)
        
        if not data:
            return None
        
        print("First attempt JSON:", data)
        
        # Apply the edit and show result to model for verification
        try:
            updated_code = self._apply_line_based_edit(
                data["insertion_point"],
                data["delete_lines"],
                new_code,
                data["indent_spaces"]
            )
            
            # Show the result to the model for verification
            numbered_result = self._add_line_numbers(updated_code)
            
            verification_prompt = (
                "Here is the result of applying your edit instructions:\n\n"
                f"{numbered_result}\n\n"
                "Does this look correct? Did the new code get inserted in the right place with proper indentation?\n"
                "If YES, respond with just: CORRECT\n"
                "If NO, provide a corrected JSON with the right insertion_point, delete_lines, and indent_spaces.\n"
                "Remember the original code had these line numbers:\n"
                f"{numbered_code}\n"
            )
            
            conversation.append({"role": "assistant", "content": json.dumps(data)})
            conversation.append({"role": "user", "content": verification_prompt})
            
            params = {
                "model": model,
                "messages": conversation,
            }
            
            if self._uses_max_completion_tokens(model):
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
            
            verification_response = client.chat.completions.create(**params)
            verification_text = verification_response.choices[0].message.content or ""
            
            print("Verification response:", verification_text)
            
            # Check if model says it's correct
            if "CORRECT" in verification_text.upper():
                print("Model verified the edit is correct")
                return data
            
            # Try to extract corrected JSON
            corrected_data = self._extract_json_from_text(verification_text)
            
            if corrected_data and self._validate_json_format(corrected_data):
                print("Second attempt JSON:", corrected_data)
                return corrected_data
            
            # If no valid correction, return original
            print("No valid correction provided, using original")
            return data
            
        except Exception as e:
            print(f"Error during verification: {e}")
            return data

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
            
            return self._extract_json_from_text(response_text)
            
        except Exception as e:
            print(f"Error calling helper: {e}")
            return None
    
    def _extract_json_from_text(self, text: str) -> dict | None:
        """Extract and parse JSON from text that may contain markdown or other content."""
        # Try to find JSON in the text
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        try:
            data = json.loads(text)
            if self._validate_json_format(data):
                return data
        except:
            pass
        
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
