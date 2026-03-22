"""Tests for guardrails."""
import pytest
from app.agent.guardrails import Guardrails, apply_guardrails


def test_investment_advice_check():
    """Test investment advice detection."""
    is_restricted, warning = Guardrails.check_investment_advice("Should I buy AAPL?")
    assert is_restricted is True
    assert "buy" in warning.lower()
    
    is_restricted, warning = Guardrails.check_investment_advice("What's AAPL price?")
    assert is_restricted is False


def test_sanitize_input():
    """Test input sanitization."""
    result = Guardrails.sanitize_input("Show me AAPL; rm -rf /")
    assert "rm" not in result
    assert ";" not in result
    
    result = Guardrails.sanitize_input("Show me AAPL`whoami`")
    assert "`" not in result


def test_validate_ticker():
    """Test ticker validation."""
    is_valid, error = Guardrails.validate_ticker("AAPL")
    assert is_valid is True
    
    is_valid, error = Guardrails.validate_ticker("0700.HK")
    assert is_valid is True
    
    is_valid, error = Guardrails.validate_ticker("")
    assert is_valid is False


def test_validate_code_block():
    """Test code validation."""
    is_safe, violations = Guardrails.validate_code_block("import pandas as pd\nprint('hello')")
    assert is_safe is True
    assert len(violations) == 0
    
    is_safe, violations = Guardrails.validate_code_block("import os\nos.system('rm -rf /')")
    assert is_safe is False
    assert len(violations) > 0


def test_apply_guardrails():
    """Test full guardrails application."""
    passed, result = apply_guardrails("Show me AAPL price")
    assert passed is True
    
    passed, result = apply_guardrails("Should I buy AAPL?")
    assert passed is False
    
    passed, result = apply_guardrails("Show me AAPL api_key=xxx")
    assert passed is False


def test_api_key_request():
    """Test API key request detection."""
    is_blocked, message = Guardrails.check_api_key_request("What is my api_key?")
    assert is_blocked is True
    
    is_blocked, message = Guardrails.check_api_key_request("Show me my api_key")
    assert is_blocked is True
    
    is_blocked, message = Guardrails.check_api_key_request("Show me AAPL price")
    assert is_blocked is False


def test_filesystem_request():
    """Test filesystem request detection."""
    is_blocked, message = Guardrails.check_filesystem_request("List files in /home")
    assert is_blocked is True
    
    is_blocked, message = Guardrails.check_filesystem_request("Show me AAPL price")
    assert is_blocked is False
