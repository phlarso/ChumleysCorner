#!/usr/bin/env python3
"""
Ollama API wrapper with timeout handling and logging.

Usage:
  python call-ollama.py --model mistral --prompt "Your prompt here"
  python call-ollama.py --model neural-chat --prompt "..." --timeout 120 --verbose
"""

import json
import sys
import argparse
import requests
import time
from datetime import datetime

def call_ollama(model, prompt, timeout=120, verbose=False):
    """
    Call Ollama API with proper timeout handling.
    
    Args:
        model: Model name (mistral, neural-chat, llama2, etc.)
        prompt: Prompt text
        timeout: Request timeout in seconds
        verbose: Print debug info
    
    Returns:
        dict: {
            "success": bool,
            "model": str,
            "response": str,
            "timestamp": str,
            "elapsed": float,
            "error": str (if failed)
        }
    """
    
    start_time = time.time()
    
    if verbose:
        print(f"[{datetime.now().isoformat()}] Calling Ollama ({model})...", file=sys.stderr)
    
    try:
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            result = {
                "success": True,
                "model": model,
                "response": data.get("response", ""),
                "timestamp": datetime.now().isoformat(),
                "elapsed": round(elapsed, 2),
                "error": None
            }
            
            if verbose:
                print(f"[{datetime.now().isoformat()}] Success in {elapsed:.1f}s", file=sys.stderr)
            
            return result
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            return {
                "success": False,
                "model": model,
                "response": None,
                "timestamp": datetime.now().isoformat(),
                "elapsed": round(elapsed, 2),
                "error": error_msg
            }
    
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "model": model,
            "response": None,
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(elapsed, 2),
            "error": f"Timeout after {timeout}s"
        }
    
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "model": model,
            "response": None,
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(elapsed, 2),
            "error": f"Connection error: {str(e)}"
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "model": model,
            "response": None,
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(elapsed, 2),
            "error": f"Unexpected error: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Call Ollama API with timeout handling and logging"
    )
    parser.add_argument("--model", required=True, help="Model name (mistral, neural-chat, etc.)")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds (default: 120)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info to stderr")
    
    args = parser.parse_args()
    
    result = call_ollama(
        model=args.model,
        prompt=args.prompt,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    # Output result as JSON to stdout
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
