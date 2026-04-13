"""Multi-layer code scanning: Semgrep + CodeQL + Bandit + LLM Review"""
import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result from a code scanner"""
    scanner: str
    findings: list[dict]
    severity_counts: dict = field(default_factory=lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0})
    scan_time: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class AggregatedScanResult:
    """Aggregated results from all scanners"""
    repository: str
    branch: str
    scanner_results: list[ScanResult]
    llm_review: dict | None = None
    total_findings: int = 0
    critical_findings: list[dict] = field(default_factory=list)


class CodeScannerConfig:
    """Configuration for code scanning tools"""
    
    def __init__(self):
        self.semgrep_rules_path = os.getenv("SEMGREP_RULES_PATH", "./rules/semgrep")
        self.codeql_databases_path = os.getenv("CODEQL_DATABASES_PATH", "./codeql_dbs")
        self.target_languages = os.getenv("TARGET_LANGUAGES", "python,javascript,java,go").split(",")
        self.semgrep_enabled = os.getenv("SEMGREP_ENABLED", "true").lower() == "true"
        self.codeql_enabled = os.getenv("CODEQL_ENABLED", "true").lower() == "true"
        self.bandit_enabled = os.getenv("BANDIT_ENABLED", "true").lower() == "true"
        self.llm_review_enabled = os.getenv("LLM_REVIEW_ENABLED", "true").lower() == "true"
        self.severity_threshold = os.getenv("SEVERITY_THRESHOLD", "medium")


class SemgrepScanner:
    """Semgrep static analysis scanner"""
    
    def __init__(self, config: CodeScannerConfig):
        self.config = config
        self.rules_path = config.semgrep_rules_path
    
    def scan(self, target_path: str) -> ScanResult:
        """Run Semgrep scan"""
        import time
        start_time = time.time()
        
        findings = []
        errors = []
        
        if not self._check_installation():
            return ScanResult(
                scanner="semgrep",
                findings=[],
                errors=["Semgrep not installed"]
            )
        
        try:
            cmd = [
                "semgrep",
                "--json",
                "--config", self.rules_path or "auto",
                "--quiet",
                target_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    findings = output.get("results", [])
                except json.JSONDecodeError:
                    errors.append("Failed to parse Semgrep JSON output")
            
            if result.stderr:
                errors.append(result.stderr)
                
        except subprocess.TimeoutExpired:
            errors.append("Semgrep scan timeout after 300s")
        except Exception as e:
            errors.append(f"Semgrep scan failed: {e}")
        
        severity_counts = self._count_severity(findings)
        
        return ScanResult(
            scanner="semgrep",
            findings=findings,
            severity_counts=severity_counts,
            scan_time=time.time() - start_time,
            errors=errors
        )
    
    def _check_installation(self) -> bool:
        """Check if Semgrep is installed"""
        try:
            subprocess.run(["semgrep", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _count_severity(self, findings: list[dict]) -> dict:
        """Count findings by severity"""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in findings:
            severity = f.get("extra", {}).get("severity", "low").lower()
            if severity in counts:
                counts[severity] += 1
        return counts


class CodeQLScanner:
    """CodeQL deep semantic analysis"""
    
    def __init__(self, config: CodeScannerConfig):
        self.config = config
        self.databases_path = config.codeql_databases_path
    
    def scan(self, target_path: str, language: str = "python") -> ScanResult:
        """Run CodeQL analysis"""
        import time
        start_time = time.time()
        
        findings = []
        errors = []
        
        if not self._check_installation():
            return ScanResult(
                scanner="codeql",
                findings=[],
                errors=["CodeQL not installed"]
            )
        
        db_path = f"{self.databases_path}/{language}_{self._sanitize_path(target_path)}"
        
        try:
            self._create_database(target_path, language, db_path)
            
            cmd = [
                "codeql", "database", "analyze",
                db_path,
                "--format=sarif-latest",
                "--output=/tmp/codeql-results.sarif"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                errors.append(result.stderr)
            else:
                findings = self._parse_sarif("/tmp/codeql-results.sarif")
                
        except subprocess.TimeoutExpired:
            errors.append("CodeQL scan timeout after 600s")
        except Exception as e:
            errors.append(f"CodeQL scan failed: {e}")
        
        severity_counts = self._count_severity(findings)
        
        return ScanResult(
            scanner="codeql",
            findings=findings,
            severity_counts=severity_counts,
            scan_time=time.time() - start_time,
            errors=errors
        )
    
    def _check_installation(self) -> bool:
        """Check if CodeQL is installed"""
        try:
            subprocess.run(["codeql", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _create_database(self, target: str, language: str, db_path: str):
        """Create CodeQL database"""
        cmd = [
            "codeql", "database", "create",
            db_path,
            "--language", language,
            "--source-root", target
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
    
    def _parse_sarif(self, sarif_path: str) -> list[dict]:
        """Parse SARIF results"""
        try:
            with open(sarif_path) as f:
                sarif = json.load(f)
            
            findings = []
            for run in sarif.get("runs", []):
                for result in run.get("results", []):
                    findings.append({
                        "rule_id": result.get("ruleId"),
                        "message": result.get("message", {}).get("text"),
                        "level": result.get("level", "warning"),
                        "locations": result.get("locations", [])
                    })
            return findings
        except Exception:
            return []
    
    def _sanitize_path(self, path: str) -> str:
        """Sanitize path for database name"""
        return path.replace("/", "_").replace("\\", "_")[:50]
    
    def _count_severity(self, findings: list[dict]) -> dict:
        """Count findings by severity"""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in findings:
            level = f.get("level", "warning")
            if "critical" in level:
                counts["critical"] += 1
            elif "error" in level:
                counts["high"] += 1
            elif "warning" in level:
                counts["medium"] += 1
            else:
                counts["low"] += 1
        return counts


class BanditScanner:
    """Bandit Python security scanner with custom rules"""
    
    def __init__(self, config: CodeScannerConfig):
        self.config = config
    
    def scan(self, target_path: str) -> ScanResult:
        """Run Bandit scan with custom rules"""
        import time
        start_time = time.time()
        
        findings = []
        errors = []
        
        if not self._check_installation():
            return ScanResult(
                scanner="bandit",
                findings=[],
                errors=["Bandit not installed"]
            )
        
        custom_rules = self._get_custom_rules()
        
        try:
            cmd = [
                "bandit",
                "-r",
                "-f", "json",
                "-ll",  # Show all confidence levels
                target_path
            ]
            
            if custom_rules:
                cmd.extend(["--exclude", ",".join(custom_rules)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    findings = output.get("results", [])
                except json.JSONDecodeError:
                    errors.append("Failed to parse Bandit JSON output")
            
            findings = self._apply_pdp_filters(findings)
            findings = self._apply_crypto_filters(findings)
            
        except subprocess.TimeoutExpired:
            errors.append("Bandit scan timeout after 120s")
        except Exception as e:
            errors.append(f"Bandit scan failed: {e}")
        
        severity_counts = self._count_severity(findings)
        
        return ScanResult(
            scanner="bandit",
            findings=findings,
            severity_counts=severity_counts,
            scan_time=time.time() - start_time,
            errors=errors
        )
    
    def _check_installation(self) -> bool:
        """Check if Bandit is installed"""
        try:
            subprocess.run(["bandit", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _get_custom_rules(self) -> list[str]:
        """Get custom PDP/Crypto rules"""
        custom = []
        
        pdp_patterns = [
            "pdp_processing",
            "personal_data",
            "biometric"
        ]
        
        crypto_patterns = [
            "weak_crypto",
            "hardcoded_key",
            "insecure_mode"
        ]
        
        return custom
    
    def _apply_pdp_filters(self, findings: list[dict]) -> list[dict]:
        """Filter for Personal Data Protection violations"""
        
        pdp_indicators = [
            "personal_data", "биометр", "иин", "инн",
            "passport", "address", "phone", "email"
        ]
        
        filtered = []
        for f in findings:
            code = f.get("code", "").lower()
            if any(indicator in code for indicator in pdp_indicators):
                f["pdp_relevant"] = True
                f["regulation"] = "ҚР Заны о персональных данных"
                filtered.append(f)
        
        return filtered
    
    def _apply_crypto_filters(self, findings: list[dict]) -> list[dict]:
        """Filter for cryptography compliance"""
        
        crypto_indicators = [
            "crypto", "encryption", "cipher", "aes", "rsa",
            "md5", "sha1", "random", "key"
        ]
        
        filtered = []
        for f in findings:
            code = f.get("code", "").lower()
            if any(indicator in code for indicator in crypto_indicators):
                f["crypto_relevant"] = True
                f["regulation"] = "КНБ стандарты криптографии"
                filtered.append(f)
        
        return filtered
    
    def _count_severity(self, findings: list[dict]) -> dict:
        """Count findings by severity"""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in findings:
            conf = f.get("issue_confidence", "LOW")
            sev = f.get("issue_severity", "LOW")
            
            if sev == "HIGH" and conf == "HIGH":
                counts["critical"] += 1
            elif sev == "HIGH":
                counts["high"] += 1
            elif sev == "MEDIUM":
                counts["medium"] += 1
            else:
                counts["low"] += 1
        return counts


class LLMReviewScanner:
    """LLM-based intelligent code review for compliance"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def review(
        self,
        code_changes: list[dict],
        critical_only: bool = False
    ) -> dict:
        """Review code changes using LLM for context and compliance"""
        
        if critical_only and len(code_changes) > 10:
            code_changes = code_changes[:10]
        
        prompt = self._build_review_prompt(code_changes)
        
        try:
            response = self.llm.client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=4096
            )
            
            return {
                "review": response.content,
                "tokens_used": response.tokens_used,
                "findings": self._parse_findings(response.content),
                "pdp_violations": self._find_pdp_violations(response.content),
                "crypto_issues": self._find_crypto_issues(response.content)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _build_review_prompt(self, changes: list[dict]) -> str:
        """Build review prompt from code changes"""
        
        changes_text = "\n\n".join([
            f"File: {c.get('file', 'unknown')}\n"
            f"Diff:\n{c.get('diff', '')}"
            for c in changes[:20]
        ])
        
        return f"""Вы эксперт по безопасности Казахстана (КНБ).
Проанализируйте следующие изменения кода на:
1. Уязвимости безопасности
2. Нарушения Закона РК о персональных данных (ПДн)
3. Соответствие криптографическим стандартам КНБ
4. Соответствие стандартам защиты информации

Изменения кода:
{changes_text}

Для каждой проблемы укажите:
- severity: critical/high/medium/low
- category: тип проблемы
- regulation: ссылка на закон/стандарт РК
- recommendation: исправление

Формат ответа: JSON массив проблем.
"""
    
    def _parse_findings(self, review_text: str) -> list[dict]:
        """Parse LLM review findings"""
        
        import re
        
        findings = []
        
        severity_pattern = r"(critical|high|medium|low)"
        category_pattern = r"(security|pdp|crypto|compliance)"
        
        return findings
    
    def _find_pdp_violations(self, review_text: str) -> list[dict]:
        """Extract PDP violations from review"""
        
        pdp_keywords = ["персональные данные", "ПДн", "ИИН", "биометр", "согласие"]
        violations = []
        
        for keyword in pdp_keywords:
            if keyword.lower() in review_text.lower():
                violations.append({"keyword": keyword, "context": "PDP relevant"})
        
        return violations
    
    def _find_crypto_issues(self, review_text: str) -> list[dict]:
        """Extract crypto issues from review"""
        
        crypto_keywords = ["шифрование", "криптография", "ключ", "AES", "RSA", "MD5"]
        issues = []
        
        for keyword in crypto_keywords:
            if keyword.lower() in review_text.lower():
                issues.append({"keyword": keyword, "context": "Crypto relevant"})
        
        return issues


class MultiLayerScanner:
    """Combined scanner using Semgrep, CodeQL, Bandit, and LLM"""
    
    def __init__(
        self,
        config: CodeScannerConfig | None = None,
        llm_client = None
    ):
        self.config = config or CodeScannerConfig()
        self.semgrep = SemgrepScanner(self.config)
        self.codeql = CodeQLScanner(self.config)
        self.bandit = BanditScanner(self.config)
        self.llm_reviewer = LLMReviewScanner(llm_client) if llm_client else None
    
    def scan_repository(
        self,
        repo_path: str,
        languages: list[str] | None = None
    ) -> AggregatedScanResult:
        """Run comprehensive scan of repository"""
        
        languages = languages or self.config.target_languages
        results = []
        
        if self.config.semgrep_enabled:
            result = self.semgrep.scan(repo_path)
            results.append(result)
            logger.info(f"Semgrep: {result.severity_counts}")
        
        if self.config.bandit_enabled and "python" in languages:
            result = self.bandit.scan(repo_path)
            results.append(result)
            logger.info(f"Bandit: {result.severity_counts}")
        
        if self.config.codeql_enabled:
            for lang in languages:
                result = self.codeql.scan(repo_path, lang)
                results.append(result)
        
        critical_findings = self._collect_critical(results)
        
        llm_review = None
        if self.config.llm_review_enabled and self.llm_reviewer and critical_findings:
            llm_review = self.llm_reviewer.review(critical_findings, critical_only=True)
        
        return AggregatedScanResult(
            repository=repo_path,
            branch="main",
            scanner_results=results,
            llm_review=llm_review,
            total_findings=sum(len(r.findings) for r in results),
            critical_findings=critical_findings
        )
    
    def _collect_critical(self, results: list[ScanResult]) -> list[dict]:
        """Collect all critical findings"""
        
        critical = []
        for result in results:
            for f in result.findings:
                severity = result.severity_counts
                if severity.get("critical", 0) > 0 or severity.get("high", 0) > 0:
                    critical.append({
                        "scanner": result.scanner,
                        **f
                    })
        
        return critical[:50]
    
    def export_json(self, result: AggregatedScanResult, output_path: str):
        """Export results to JSON"""
        
        export_data = {
            "repository": result.repository,
            "branch": result.branch,
            "total_findings": result.total_findings,
            "scanners": [
                {
                    "name": r.scanner,
                    "findings_count": len(r.findings),
                    "severity": r.severity_counts,
                    "scan_time": r.scan_time
                }
                for r in result.scanner_results
            ],
            "critical_findings": result.critical_findings,
            "llm_review": result.llm_review
        }
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to {output_path}")


if __name__ == "__main__":
    from src.agents.llm_client import LocalLLMClient
    
    config = CodeScannerConfig()
    llm = LocalLLMClient()
    
    scanner = MultiLayerScanner(config, llm)
    
    result = scanner.scan_repository("./src")
    print(f"Total findings: {result.total_findings}")
    print(f"Critical: {len(result.critical_findings)}")
    
    if result.llm_review:
        print(f"LLM Review: {result.llm_review.get('review', 'N/A')[:200]}")