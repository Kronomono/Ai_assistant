import os
import json
import threading
import time
from typing import List, Dict, Any
from queue import Queue
from datetime import datetime, timedelta
from hyperdb import HyperDB, get_embedding
from functools import lru_cache
import gzip
import pickle

class Memory:
    def __init__(self, embedding_model=None, max_connections=5):
        self.memory_dir = os.path.join(os.path.dirname(__file__), 'memory')
        os.makedirs(self.memory_dir, exist_ok=True)
        
        self.metadata_file = os.path.join(self.memory_dir, 'metadata.json')
        self.transaction_log = os.path.join(self.memory_dir, 'transactions.log')
        self.embedding_function = embedding_model or get_embedding
        
        self.metadata = self._load_metadata()
        self.connection_pool = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        for _ in range(max_connections):
            db = HyperDB(embedding_function=self.embedding_function)
            self.connection_pool.put(db)

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'files': {}}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def _get_db_connection(self):
        return self.connection_pool.get()

    def _release_db_connection(self, db):
        self.connection_pool.put(db)

    def _log_transaction(self, operation, details):
        with open(self.transaction_log, 'a') as f:
            f.write(f"{time.time()},{operation},{json.dumps(details)}\n")

    def _get_db_file_for_date(self, date):
        date_str = date.strftime("%Y-%m-%d")
        if date_str not in self.metadata['files']:
            file_name = f"memory_{date_str}.gz"
            self.metadata['files'][date_str] = os.path.join(self.memory_dir, file_name)
            self._save_metadata()
        return self.metadata['files'][date_str]

    def add_conversation(self, conversation: List[Dict[str, str]]):
        db = self._get_db_connection()
        try:
            full_conversation = " ".join([f"{msg['role']} ({msg['timestamp']}): {msg['content']}" for msg in conversation])
            date = datetime.strptime(conversation[-1]['timestamp'], "%Y-%m-%d %H:%M:%S").date()
            
            file_path = self._get_db_file_for_date(date)
            if os.path.exists(file_path):
                db.load(file_path)
            
            db.add({"conversation": full_conversation, "date": str(date), "timestamp": conversation[-1]['timestamp']})
            db.save(file_path)
            
            self._log_transaction('add', {'date': str(date), 'timestamp': conversation[-1]['timestamp']})
        except Exception as e:
            self._log_transaction('error', {'operation': 'add', 'error': str(e)})
            raise
        finally:
            self._release_db_connection(db)

    def query_memory(self, query: str, top_k: int = 5, date_range: tuple = None) -> List[Dict[str, Any]]:
        results = []
        db_files = self._get_relevant_db_files(date_range)
        
        for file_path in db_files:
            db = self._get_db_connection()
            try:
                db.load(file_path)
                results.extend(db.query(query, top_k=top_k))
            except Exception as e:
                self._log_transaction('error', {'operation': 'query', 'file': file_path, 'error': str(e)})
            finally:
                self._release_db_connection(db)
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def _get_relevant_db_files(self, date_range=None):
        if date_range:
            start_date, end_date = date_range
            return [self.metadata['files'][date_str] for date_str in self.metadata['files']
                    if start_date <= datetime.strptime(date_str, "%Y-%m-%d").date() <= end_date]
        else:
            return list(self.metadata['files'].values())

    @lru_cache(maxsize=100)
    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        relevant_memories = self.query_memory(query, top_k=top_k)
        if not relevant_memories:
            return ""
        context = "\n\n".join([f"Past conversation (timestamp: {m[0]['timestamp']}, similarity: {float(m[1]):.2f}):\n{m[0]['conversation']}" for m in relevant_memories])
        return context

    def clear_memory(self):
        with self.lock:
            for file_path in self.metadata['files'].values():
                if os.path.exists(file_path):
                    os.remove(file_path)
            self.metadata['files'] = {}
            self._save_metadata()
            self._log_transaction('clear', {})

    def get_all_conversations(self) -> List[Dict[str, str]]:
        all_conversations = []
        for file_path in self.metadata['files'].values():
            db = self._get_db_connection()
            try:
                db.load(file_path)
                all_conversations.extend(db.documents)
            except Exception as e:
                self._log_transaction('error', {'operation': 'get_all', 'file': file_path, 'error': str(e)})
            finally:
                self._release_db_connection(db)
        return all_conversations

    def delete_conversation(self, date: datetime.date, index: int):
        file_path = self._get_db_file_for_date(date)
        db = self._get_db_connection()
        try:
            db.load(file_path)
            if 0 <= index < len(db.documents):
                db.remove_document(index)
                db.save(file_path)
                self._log_transaction('delete', {'date': str(date), 'index': index})
            else:
                raise IndexError("Invalid conversation index")
        except Exception as e:
            self._log_transaction('error', {'operation': 'delete', 'file': file_path, 'error': str(e)})
            raise
        finally:
            self._release_db_connection(db)

    def update_conversation(self, date: datetime.date, index: int, new_conversation: Dict[str, str]):
        file_path = self._get_db_file_for_date(date)
        db = self._get_db_connection()
        try:
            db.load(file_path)
            if 0 <= index < len(db.documents):
                db.documents[index] = new_conversation
                db.vectors[index] = self.embedding_function([new_conversation])[0]
                db.save(file_path)
                self._log_transaction('update', {'date': str(date), 'index': index})
            else:
                raise IndexError("Invalid conversation index")
        except Exception as e:
            self._log_transaction('error', {'operation': 'update', 'file': file_path, 'error': str(e)})
            raise
        finally:
            self._release_db_connection(db)