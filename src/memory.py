import os
from typing import List, Dict, Any
from hyperdb import HyperDB, get_embedding

class Memory:
    def __init__(self, embedding_model=None):
        self.db = HyperDB(embedding_function=embedding_model or get_embedding)
        self.current_conversation: List[Dict[str, Any]] = []
        self.memory_dir = os.path.join(os.path.dirname(__file__), 'memory')
        
        # Ensure the memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)

    def add_to_conversation(self, role: str, content: str):
        """Add a new message to the current conversation."""
        self.current_conversation.append({"role": role, "content": content})

    def save_current_conversation(self):
        """Save the current conversation to the vector database."""
        if not self.current_conversation:
            return

        # Combine all messages into a single string
        full_conversation = " ".join([f"{msg['role']}: {msg['content']}" for msg in self.current_conversation])

        # Add to the database
        self.db.add({"conversation": full_conversation})

        # Clear the current conversation
        self.current_conversation = []

    def query_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the memory for relevant past conversations."""
        results = self.db.query(query, top_k=top_k)
        return [{"conversation": r[0]["conversation"], "similarity": float(r[1])} for r in results]

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Get relevant context from past conversations for the given query."""
        relevant_memories = self.query_memory(query, top_k=top_k)
        if not relevant_memories:
            return ""  # Return an empty string if no relevant memories are found
        context = "\n\n".join([f"Past conversation (similarity: {m['similarity']:.2f}):\n{m['conversation']}" for m in relevant_memories])
        return context

    def clear_memory(self):
        """Clear all stored memories."""
        self.db = HyperDB(embedding_function=self.db.embedding_function)
        self.current_conversation = []

    def save_to_file(self, filename: str):
        """Save the current state of the memory to a file."""
        full_path = os.path.join(self.memory_dir, filename)
        self.db.save(full_path)

    def load_from_file(self, filename: str):
        """Load the state of the memory from a file."""
        full_path = os.path.join(self.memory_dir, filename)
        if os.path.exists(full_path):
            self.db.load(full_path)
        else:
            print(f"Memory file {filename} not found. Starting with empty memory.")

    def append_to_file(self, filename: str):
        """Append current memory to existing file or create new if not exists."""
        full_path = os.path.join(self.memory_dir, filename)
        if os.path.exists(full_path):
            existing_db = HyperDB(embedding_function=self.db.embedding_function)
            existing_db.load(full_path)
            # Append new conversations to existing ones
            existing_db.add_documents(self.db.documents)
            existing_db.save(full_path)
        else:
            # If file doesn't exist, just save current memory
            self.save_to_file(filename)