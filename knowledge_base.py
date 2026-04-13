"""Knowledge Base with PostgreSQL + pgvector for Kazakhstan regulations"""
import os
import hashlib
from datetime import datetime
from typing import Generator, Optional
from dataclasses import dataclass
import logging

import psycopg2
from psycopg2 import sql
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Chunk of a document with metadata"""
    id: str
    content: str
    metadata: dict
    embedding: list[float] | None = None
    law_number: str | None = None
    effective_date: datetime | None = None
    category: str | None = None


@dataclass
class SearchResult:
    """Search result from knowledge base"""
    content: str
    metadata: dict
    score: float
    law_number: str | None


class KnowledgeBaseConfig:
    """Knowledge base configuration"""
    
    def __init__(self):
        self.host = os.getenv("KB_DB_HOST", "localhost")
        self.port = int(os.getenv("KB_DB_PORT", "5432"))
        self.database = os.getenv("KB_DATABASE", "knowledge_base")
        self.user = os.getenv("KB_USER", "postgres")
        self.password = os.getenv("KB_PASSWORD", "")
        self.vector_dimension = int(os.getenv("VECTOR_DIMENSION", "1536"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "128"))


class KnowledgeBase:
    """PostgreSQL + pgvector knowledge base for Kazakhstan regulations"""
    
    def __init__(self, config: KnowledgeBaseConfig | None = None):
        self.config = config or KnowledgeBaseConfig()
        self.conn = None
        self._connect()
        self._init_schema()
    
    def _connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            self.conn.autocommit = True
            logger.info(f"Connected to knowledge base at {self.config.host}")
        except Exception as e:
            logger.error(f"Failed to connect to knowledge base: {e}")
            raise
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    law_number VARCHAR(100),
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category VARCHAR(100),
                    authority VARCHAR(200),
                    effective_date DATE,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    chunk_index INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(%s),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """, (self.config.vector_dimension,))
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_law_number 
                ON documents(law_number)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_category 
                ON documents(category)
            """)
    
    def add_document(
        self,
        title: str,
        content: str,
        law_number: str | None = None,
        category: str | None = None,
        authority: str | None = None,
        effective_date: datetime | None = None
    ) -> int:
        """Add a document to the knowledge base"""
        
        doc_id = None
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (law_number, title, content, category, authority, effective_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (law_number, title, content, category, authority, effective_date))
            doc_id = cur.fetchone()[0]
            
            chunks = self._create_chunks(content)
            for idx, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk)
                cur.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (doc_id, idx, chunk, embedding))
        
        logger.info(f"Added document {doc_id}: {title}")
        return doc_id
    
    def _create_chunks(self, text: str) -> list[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.config.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.config.chunk_size - self.config.chunk_overlap
        
        return chunks
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using local model"""
        import requests
        
        try:
            response = requests.post(
                os.getenv("EMBEDDING_ENDPOINT", "http://localhost:8081/embed"),
                json={"text": text},
                timeout=30
            )
            return response.json().get("embedding", [0.0] * self.config.vector_dimension)
        except Exception:
            logger.warning("Embedding service unavailable, using zero vector")
            return [0.0] * self.config.vector_dimension
    
    def hybrid_search(
        self,
        query: str,
        law_number_filter: str | None = None,
        category_filter: str | None = None,
        limit: int = 5
    ) -> list[SearchResult]:
        """Hybrid search: vector + keyword + filters"""
        
        query_embedding = self._get_embedding(query)
        
        with self.conn.cursor() as cur:
            vector_query = """
                SELECT dc.content, d.law_number, d.category, d.authority,
                       1 - (dc.embedding <=> %s::vector) as score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE 1=1
            """
            params = [query_embedding]
            
            if law_number_filter:
                vector_query += " AND d.law_number = %s"
                params.append(law_number_filter)
            
            if category_filter:
                vector_query += " AND d.category = %s"
                params.append(category_filter)
            
            vector_query += f" ORDER BY score DESC LIMIT {limit}"
            
            cur.execute(vector_query, params)
            results = []
            
            for row in cur.fetchall():
                results.append(SearchResult(
                    content=row[0],
                    metadata={"law_number": row[1], "category": row[2], "authority": row[3]},
                    score=float(row[4]),
                    law_number=row[1]
                ))
            
            return results
    
    def get_by_law_number(self, law_number: str) -> list[dict]:
        """Get all documents for a specific law number"""
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, law_number, title, content, category, effective_date, version
                FROM documents
                WHERE law_number = %s
                ORDER BY version DESC
            """, (law_number,))
            
            return [
                {
                    "id": row[0],
                    "law_number": row[1],
                    "title": row[2],
                    "content": row[3],
                    "category": row[4],
                    "effective_date": row[5],
                    "version": row[6]
                }
                for row in cur.fetchall()
            ]
    
    def get_latest_version(self, law_number: str) -> dict | None:
        """Get the latest version of a document"""
        
        versions = self.get_by_law_number(law_number)
        return versions[0] if versions else None
    
    def update_document(
        self,
        law_number: str,
        new_content: str,
        new_title: str | None = None
    ) -> int:
        """Update document with new version"""
        
        latest = self.get_latest_version(law_number)
        if latest:
            new_version = latest["version"] + 1
        else:
            new_version = 1
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (law_number, title, content, version)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (law_number, new_title or latest["title"], new_content, new_version))
            doc_id = cur.fetchone()[0]
            
            chunks = self._create_chunks(new_content)
            for idx, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk)
                cur.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (doc_id, idx, chunk, embedding))
        
        logger.info(f"Updated document {law_number} to version {new_version}")
        return doc_id
    
    def search_by_authority(self, authority: str, limit: int = 10) -> list[dict]:
        """Search documents by authority (e.g., 'КНБ РК', 'ҚР Үкіметі')"""
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, law_number, title, category, effective_date
                FROM documents
                WHERE authority ILIKE %s
                ORDER BY effective_date DESC
                LIMIT %s
            """, (f"%{authority}%", limit))
            
            return [
                {
                    "id": row[0],
                    "law_number": row[1],
                    "title": row[2],
                    "category": row[3],
                    "effective_date": row[4]
                }
                for row in cur.fetchall()
            ]
    
    def get_statistics(self) -> dict:
        """Get knowledge base statistics"""
        
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            chunk_count = cur.fetchone()[0]
            
            cur.execute("""
                SELECT category, COUNT(*) 
                FROM documents 
                GROUP BY category
            """)
            categories = {row[0]: row[1] for row in cur.fetchall()}
            
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "categories": categories
            }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class KazakhstanRegulations:
    """Pre-loaded Kazakhstan regulation categories"""
    
    CATEGORIES = {
        "pdp": "Защита персональных данных",
        "crypto": "Криптографическая защита",
        "info_security": "Информационная безопасность",
        "state_secrets": "Государственные секреты",
        "privacy": "Конфиденциальность",
    }
    
    AUTHORITIES = {
        "knb": "Комитет национальной безопасности РК",
        "government": "Правительство РК",
        " parliament": "Парламент РК",
        "president": "Президент РК",
        "mvd": "Министерство внутренних дел РК",
    }


if __name__ == "__main__":
    kb = KnowledgeBase()
    
    kb.add_document(
        title="Закон РК о персональных данных",
        content="""Закон Республики Казахстан о персональных данных и их защите
Статья 1. Основные понятия
1. Персональные данные - сведения, относящиеся к определенному или определяемому субъекту персональных данных...
Статья 2. Субъекты персональных данных
Субъектом персональных данных является физическое лицо...
Статья 5. Обработка персональных данных
Обработка персональных данных осуществляется с согласия субъекта...
""",
        law_number="ҚР Заны 2022-ХІV",
        category="pdp",
        authority="Парламент РК",
        effective_date=datetime(2022, 1, 1)
    )
    
    results = kb.hybrid_search("обработка персональных данных", category_filter="pdp")
    
    for r in results:
        print(f"[{r.score:.3f}] {r.metadata.get('law_number')}: {r.content[:100]}...")
    
    print(kb.get_statistics())