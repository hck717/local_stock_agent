"""Safe code execution sandbox."""
import ast
import io
import sys
import traceback
import signal
from typing import Optional, Dict, Any
from contextlib import redirect_stdout, redirect_stderr

from app.config import config


class ExecutionTimeout(Exception):
    """Exception raised when code execution times out."""
    pass


def _timeout_handler(signum, frame):
    raise ExecutionTimeout("Code execution timed out")


ALLOWED_IMPORTS = set(config.APPROVED_IMPORTS)

DANGEROUS_PATTERNS = [
    "os.system", "subprocess", "socket", "requests", "urllib",
    "eval", "exec", "compile", "__import__",
    "open(", "file(", "input(",
    "rm ", "rmdir", "unlink", "delete",
    "shutil.rmtree", "os.remove",
    "cursor.execute", "connection.",
    "os.environ", "getenv", "setenv",
    "popen", "pty", "tty",
]


def validate_code(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python code for safety.
    
    Args:
        code: Python code to validate
    
    Returns:
        tuple of (is_valid, error_message)
    """
    code_lower = code.lower()
    
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in code_lower:
            return False, f"Blocked dangerous pattern: {pattern}"
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_IMPORTS:
                        if not any(alias.name.startswith(f"{p}.") for p in ALLOWED_IMPORTS):
                            return False, f"Import '{alias.name}' not allowed"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in ALLOWED_IMPORTS:
                    if not any(node.module.startswith(f"{p}.") for p in ALLOWED_IMPORTS):
                        return False, f"Import from '{node.module}' not allowed"
            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile", "open", "input"]:
                        return False, f"Call to '{node.func.id}' not allowed"
    
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    return True, None


def run_safe_analysis(
    code: str,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute Python analysis code in a safe sandbox.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
    
    Returns:
        dict with execution results
    """
    timeout = timeout or config.MAX_CODE_EXECUTION_TIME
    
    is_valid, error = validate_code(code)
    if not is_valid:
        return {
            "status": "UNSAFE_CODE_BLOCKED",
            "message": error,
            "output": None,
            "artifacts": []
        }
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    local_vars = {}
    artifacts = []
    
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, {"__builtins__": __builtins__}, local_vars)
        
        signal.alarm(0)
        
        for key, value in local_vars.items():
            if key.startswith('_'):
                continue
            if hasattr(value, 'to_html') or hasattr(value, 'write_html'):
                try:
                    html_path = config.OUTPUTS_DIR / f"analysis_{key}.html"
                    value.write_html(str(html_path))
                    artifacts.append(str(html_path))
                except:
                    pass
        
        return {
            "status": "success",
            "output": stdout_capture.getvalue(),
            "errors": stderr_capture.getvalue(),
            "local_vars": {k: str(v) for k, v in local_vars.items() if not k.startswith('_')},
            "artifacts": artifacts
        }
    
    except ExecutionTimeout:
        signal.alarm(0)
        return {
            "status": "SANDBOX_TIMEOUT",
            "message": f"Code execution exceeded {timeout} seconds",
            "output": stdout_capture.getvalue(),
            "artifacts": []
        }
    
    except Exception as e:
        signal.alarm(0)
        return {
            "status": "ERROR",
            "message": f"Execution error: {str(e)}\n{traceback.format_exc()}",
            "output": stdout_capture.getvalue(),
            "errors": stderr_capture.getvalue(),
            "artifacts": []
        }
    
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def generate_analysis_code(
    task: str,
    ticker: str,
    indicators: Optional[list[str]] = None
) -> str:
    """
    Generate Python analysis code based on task description.
    
    Args:
        task: Description of analysis to perform
        ticker: Stock ticker symbol
        indicators: List of technical indicators
    
    Returns:
        Python code string
    """
    task = task.lower()
    
    if "bollinger" in task:
        return f'''import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

ticker = "{ticker}"
data = yf.Ticker(ticker).history(period="6mo")

close = data['Close']
sma20 = close.rolling(window=20).mean()
std20 = close.rolling(window=20).std()

data['BB_Upper'] = sma20 + (std20 * 2)
data['BB_Middle'] = sma20
data['BB_Lower'] = sma20 - (std20 * 2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=close, mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='Upper Band', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='Middle Band', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='Lower Band', line=dict(dash='dash')))
fig.update_layout(title=f'{{ticker}} Bollinger Bands', template='plotly_dark')
fig.write_html("outputs/{ticker}_bollinger_bands.html")
print("Chart saved to outputs/{ticker}_bollinger_bands.html")
'''
    
    elif "sharpe" in task or "risk-adjusted" in task:
        return f'''import yfinance as yf
import pandas as pd
import numpy as np

ticker = "{ticker}"
data = yf.Ticker(ticker).history(period="1y")

returns = data['Close'].pct_change().dropna()

sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
annual_return = returns.mean() * 252 * 100
annual_vol = returns.std() * np.sqrt(252) * 100

print(f"Sharpe Ratio: {{sharpe_ratio:.2f}}")
print(f"Annual Return: {{annual_return:.2f}}%")
print(f"Annual Volatility: {{annual_vol:.2f}}%")
'''
    
    elif "backtest" in task or "sma crossover" in task:
        return f'''import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

ticker = "{ticker}"
data = yf.Ticker(ticker).history(period="2y")

data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

data['Signal'] = 0
data.loc[data['SMA_20'] > data['SMA_50'], 'Signal'] = 1
data.loc[data['SMA_20'] < data['SMA_50'], 'Signal'] = -1
data['Position'] = data['Signal'].diff()

data['Strategy_Return'] = data['Close'].pct_change() * data['Signal'].shift(1)
data['Buy_Hold_Return'] = data['Close'].pct_change()

strategy_return = (1 + data['Strategy_Return'].dropna()).cumprod().iloc[-1] - 1
buy_hold_return = (1 + data['Buy_Hold_Return'].dropna()).cumprod().iloc[-1] - 1

print(f"Strategy Return: {{strategy_return*100:.2f}}%")
print(f"Buy & Hold Return: {{buy_hold_return*100:.2f}}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))

buy_signals = data[data['Position'] == 2]
sell_signals = data[data['Position'] == -2]

fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)))

fig.update_layout(title=f'{{ticker}} SMA Crossover Backtest', template='plotly_dark')
fig.write_html("outputs/{{ticker}}_sma_crossover.html")
'''
    
    else:
        indicators_code = ""
        if indicators:
            ind_list = ", ".join([f'"{ind}"' for ind in indicators])
            indicators_code = f"""
indicators = [{ind_list}]
for ind in indicators:
    if ind in data.columns:
        print(f"{{ind}}: {{data[ind].iloc[-1]:.2f}}")
"""
        
        return f'''import yfinance as yf
import pandas as pd

ticker = "{ticker}"
data = yf.Ticker(ticker).history(period="6mo")

print(f"Analysis for {{ticker}}")
print(f"Current Price: {{data['Close'].iloc[-1]:.2f}}")
print(f"6-Month Return: {{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}}%")
{indicators_code}
'''
