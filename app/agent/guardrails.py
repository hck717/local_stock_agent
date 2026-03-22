"""Guardrails for safe agent operation."""
import re
from typing import Tuple, Optional, List


RESTRICTED_PATTERNS = [
    (r"\bbuy\b", "We don't provide buy/sell recommendations"),
    (r"\bsell\b", "We don't provide buy/sell recommendations"),
    (r"\bhold\b", "We don't provide buy/sell recommendations"),
    (r"\bshould\s+(i|you)\s+buy", "We don't provide investment recommendations"),
    (r"\bshould\s+(i|you)\s+sell", "We don't provide investment recommendations"),
    (r"\binvest\s+(in|into)", "We don't provide investment advice"),
    (r"\bput\s+money\s+(in|into)", "We don't provide investment advice"),
    (r"\bmy\s+(money|portfolio)", "We don't provide personalized financial advice"),
    (r"\bbest\s+(stock|investment)", "We don't recommend specific investments"),
    (r"\bworst\s+(stock|investment)", "We don't recommend specific investments"),
    (r"\btrade\b", "This is not a trading platform"),
    (r"\brobot\s+(advisor|adviser)", "This is not a robo-advisor"),
    (r"\bfinancial\s+planner", "This is not a financial planning service"),
    (r"\btax\s+advice", "Please consult a tax professional"),
    (r"\btaxes?\s+implication", "Please consult a tax professional"),
]


INPUT_SANITIZATION_PATTERNS = [
    (r"[;&|`$]", ""),
    (r"\brm\s+", ""),
    (r"\bdel\s+", ""),
    (r"\bdrop\s+", ""),
    (r"\btruncate\s+", ""),
    (r"\bexec\s*\(", ""),
    (r"\bsystem\s*\(", ""),
]


DANGEROUS_CODE_PATTERNS = [
    "os.system",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "eval(",
    "exec(",
    "compile(",
    "__import__",
    "open(",
    "file(",
    "input(",
    "rm -",
    "rmdir",
    "unlink",
    "shutil.rmtree",
    "os.remove",
    "cursor.execute",
    "os.environ",
]


class Guardrails:
    """Guardrails for the stock analysis agent."""
    
    @staticmethod
    def check_investment_advice(text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains investment advice requests.
        
        Returns:
            tuple of (is_restricted, warning_message)
        """
        text_lower = text.lower()
        
        for pattern, message in RESTRICTED_PATTERNS:
            if re.search(pattern, text_lower):
                return True, message
        
        return False, None
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitize user input.
        
        Args:
            text: Raw user input
        
        Returns:
            Sanitized text
        """
        result = text
        
        for pattern, replacement in INPUT_SANITIZATION_PATTERNS:
            result = re.sub(pattern, replacement, result)
        
        result = re.sub(r'\s+', ' ', result).strip()
        
        if len(result) > 1000:
            result = result[:1000]
        
        return result
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a ticker symbol.
        
        Returns:
            tuple of (is_valid, error_message)
        """
        if not ticker:
            return False, "Ticker cannot be empty"
        
        ticker = ticker.strip().upper()
        
        if len(ticker) > 15:
            return False, "Ticker too long"
        
        if not re.match(r'^[A-Z0-9\.\-\^]+$', ticker):
            return False, "Ticker contains invalid characters"
        
        return True, None
    
    @staticmethod
    def validate_code_block(code: str) -> Tuple[bool, List[str]]:
        """
        Validate code for dangerous patterns.
        
        Returns:
            tuple of (is_safe, list_of_violations)
        """
        violations = []
        
        for pattern in DANGEROUS_CODE_PATTERNS:
            if pattern in code:
                violations.append(f"Blocked dangerous pattern: {pattern}")
        
        if len(code) > 5000:
            violations.append("Code too long (max 5000 characters)")
        
        lines = code.split('\n')
        if len(lines) > 200:
            violations.append("Too many lines (max 200)")
        
        return len(violations) == 0, violations
    
    @staticmethod
    def check_api_key_request(text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if user is requesting API keys or credentials.
        
        Returns:
            tuple of (is_blocked, message)
        """
        text_lower = text.lower()
        
        patterns = [
            r"api[_-]?key",
            r"secret",
            r"password",
            r"credential",
            r"token",
            r"auth",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True, "We cannot provide API keys or credentials."
        
        return False, None
    
    @staticmethod
    def check_filesystem_request(text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if user is requesting filesystem access.
        
        Returns:
            tuple of (is_blocked, message)
        """
        text_lower = text.lower()
        
        patterns = [
            r"list\s+files",
            r"read\s+.*file",
            r"write\s+.*file",
            r"delete\s+.*file",
            r"download\s+.*file",
            r"ssh\s+",
            r"ftp\s+",
            r"sftp\s+",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True, "Filesystem access is not permitted for security reasons."
        
        return False, None
    
    @staticmethod
    def check_rate_limit(user_id: str = "default") -> Tuple[bool, Optional[str]]:
        """
        Check if user has exceeded rate limits.
        
        Returns:
            tuple of (is_allowed, message)
        """
        return True, None
    
    @staticmethod
    def add_disclaimer(text: str, intent: str = "general") -> str:
        """
        Add appropriate disclaimer to response.
        
        Args:
            text: Response text
            intent: Type of analysis
        
        Returns:
            Text with disclaimer appended
        """
        disclaimers = {
            "general": "\n\n⚠️ This analysis is for informational purposes only.",
            "technical": "\n\n⚠️ Technical indicators are not guarantees of future performance.",
            "fundamental": "\n\n⚠️ Fundamental analysis should be combined with other research.",
            "comparison": "\n\n⚠️ Past performance comparisons do not predict future results.",
        }
        
        disclaimer = disclaimers.get(intent, disclaimers["general"])
        disclaimer += "\nYahoo Finance data may be delayed. Not suitable for real-time trading."
        
        return text + disclaimer


def apply_guardrails(text: str) -> Tuple[bool, str]:
    """
    Apply all guardrails to input text.
    
    Args:
        text: User input
    
    Returns:
        tuple of (passed, sanitized_text_or_error)
    """
    sanitized = Guardrails.sanitize_input(text)
    
    is_restricted, warning = Guardrails.check_investment_advice(sanitized)
    if is_restricted:
        return False, warning
    
    is_blocked, message = Guardrails.check_api_key_request(sanitized)
    if is_blocked:
        return False, message
    
    is_blocked, message = Guardrails.check_filesystem_request(sanitized)
    if is_blocked:
        return False, message
    
    return True, sanitized
