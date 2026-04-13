"""Kazakhstan Security Agent System - Main Entry Point"""
import os
import asyncio
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from environment and config files"""
    config = {
        "llm_endpoint": os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1"),
        "llm_model": os.getenv("LLM_MODEL", "qwen2.5-72b-instruct"),
        "kb_db_host": os.getenv("KB_DB_HOST", "localhost"),
        "kb_db_port": int(os.getenv("KB_DB_PORT", "5432")),
        "kb_database": os.getenv("KB_DATABASE", "knowledge_base"),
        "data_residency": os.getenv("DATA_RESIDENCY", "KZ"),
        "prohibited_providers": ["openai", "anthropic", "grok", "deepseek"]
    }
    return config


def verify_data_residency(config: dict):
    """Verify all external providers are prohibited"""
    endpoint = config["llm_endpoint"].lower()
    
    for provider in config["prohibited_providers"]:
        if provider in endpoint:
            raise ValueError(
                f"PROHIBITED: External provider '{provider}' detected in endpoint. "
                f"All requests must stay within Kazakhstan (KZ)."
            )
    
    logger.info("Data residency check: PASSED - all requests will stay in KZ")


async def run_code_scan(repo_path: str):
    """Run code scanning pipeline"""
    from src.scanners.code_scanner import MultiLayerScanner, CodeScannerConfig
    from src.agents.llm_client import LocalLLMClient
    
    logger.info(f"Starting code scan for: {repo_path}")
    
    config = CodeScannerConfig()
    llm = LocalLLMClient()
    scanner = MultiLayerScanner(config, llm)
    
    result = scanner.scan_repository(repo_path)
    
    logger.info(f"Scan complete: {result.total_findings} findings")
    logger.info(f"Critical: {len(result.critical_findings)}")
    
    return result


async def run_knowledge_query(query: str):
    """Query knowledge base"""
    from src.knowledge_base.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
    
    logger.info(f"Querying knowledge base: {query}")
    
    kb = KnowledgeBase()
    results = kb.hybrid_search(query, limit=5)
    
    logger.info(f"Found {len(results)} results")
    
    return results


async def run_compliance_check(document_text: str, regulation_type: str):
    """Run compliance check using LLM"""
    from src.agents.llm_client import LocalLLMClient
    
    logger.info(f"Checking compliance for regulation: {regulation_type}")
    
    llm = LocalLLMClient()
    wrapper = llm
    
    result = wrapper.check_regulatory_compliance(document_text, regulation_type)
    
    return result


def main():
    """Main entry point"""
    config = load_config()
    
    logger.info("=" * 60)
    logger.info("Kazakhstan Security Agent System")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    verify_data_residency(config)
    
    logger.info("System ready. Use the API or run specific tasks.")


if __name__ == "__main__":
    main()