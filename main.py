# main.py
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
import asyncio
import inspect
import json
import re
import ast
from typing import Dict, Any
from agent_core import handle_query

from starlette.datastructures import UploadFile as StarletteUploadFile

app = FastAPI()

async def _extract_question_and_files(request: Request):
    """
    Return (question_text, files_dict)
    files_dict: { field_name: { "filename": ..., "content_type": ..., "bytes": b"..."} }
    """
    form = await request.form()
    files: Dict[str, Dict[str, Any]] = {}
    question_text = None
    preferred_field_names = {"questions.txt", "question", "questions", "question.txt"}

    print(f"Form data received: {list(form.keys())}")  # Debug: Log form keys

    for field_name, value in form.items():
        print(f"Processing field: {field_name}, type: {type(value)}")  # Debug
        # Use the StarletteUploadFile class for the isinstance check
        if isinstance(value, StarletteUploadFile):
            raw = await value.read()
            print(f"File: {value.filename}, content_type: {value.content_type}, size: {len(raw)} bytes")  # Debug
            print(f"Raw content (first 100 bytes): {raw[:100]!r}")  # Debug: Log raw content
            files[field_name] = {
                "filename": value.filename,
                "content_type": value.content_type,
                "bytes": raw,
            }
            if not raw:
                print(f"Error: File {field_name} is empty")  # Debug
                raise ValueError(f"File {field_name} is empty")
            if field_name.strip().lower() in preferred_field_names:
                try:
                    question_text = raw.decode("utf-8").strip()
                    print(f"Extracted question text: {question_text}")  # Debug
                    break  # Exit loop after extracting question
                except UnicodeDecodeError as e:
                    print(f"Failed to decode {field_name} as UTF-8: {e}")  # Debug
                    raise ValueError(f"Failed to decode {field_name} as UTF-8: {e}")
                except Exception as e:
                    print(f"Error processing {field_name}: {e}")  # Debug
                    raise ValueError(f"Error processing {field_name}: {e}")
            # Additional heuristics for other text files (if needed)
            elif question_text is None and value.filename and value.filename.lower().endswith(".txt"):
                try:
                    question_text = raw.decode("utf-8").strip()
                    print(f"Extracted question text from filename heuristic: {question_text}")  # Debug
                except Exception as e:
                    print(f"Filename heuristic failed for {value.filename}: {e}")  # Debug
            elif question_text is None and value.content_type and value.content_type.startswith("text"):
                try:
                    question_text = raw.decode("utf-8").strip()
                    print(f"Extracted question text from content_type heuristic: {question_text}")  # Debug
                except Exception as e:
                    print(f"Content_type heuristic failed for {value.content_type}: {e}")  # Debug
        else:
            print(f"Non-file field: {field_name}, value: {value}")  # Debug
            # Skip question extraction for non-file fields

    if not question_text:
        raise ValueError(f"No question text found in questions.txt. Form fields: {list(form.keys())}")
    return question_text, files

def _call_handle_query_sync_or_async(question: str, files: Dict[str, Any]):
    print(f"Calling handle_query with question: {question}")  # Debug
    sig = inspect.signature(handle_query)
    params = list(sig.parameters.values())
    accepts_files = len(params) >= 2

    if inspect.iscoroutinefunction(handle_query):
        if accepts_files:
            return handle_query(question, files)
        else:
            return handle_query(question)
    else:
        loop = asyncio.get_running_loop()
        if accepts_files:
            return loop.run_in_executor(None, handle_query, question, files)
        else:
            return loop.run_in_executor(None, handle_query, question)

def _try_extract_json_from_text(s: str):
    if not isinstance(s, str):
        return None
    s_strip = s.strip()
    
    # First, try to parse as valid JSON
    try:
        return json.loads(s_strip)
    except (json.JSONDecodeError, TypeError):
        pass

    # If that fails, try to parse it as a Python literal (e.g., dictionary with single quotes)
    try:
        candidate = ast.literal_eval(s_strip)
        if isinstance(candidate, (dict, list)):
            return candidate
    except (ValueError, SyntaxError):
        pass
    
    # Finally, try the regex-based extraction from the original code as a last resort
    patterns = [
        r"(\[\s*[\s\S]*?\])",  # array
        r"(\{\s*[\s\S]*?\})",  # object
    ]
    for pat in patterns:
        m = re.search(pat, s, re.DOTALL)
        if m:
            candidate = m.group(1).strip().strip("`")
            try:
                # Try JSON again with the extracted content
                return json.loads(candidate)
            except Exception:
                # Try ast.literal_eval with the extracted content
                try:
                    return ast.literal_eval(candidate)
                except Exception:
                    pass
    return None

@app.post("/api/")
async def run_query(request: Request):
    try:
        question_text, files = await _extract_question_and_files(request)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        coro_or_future = _call_handle_query_sync_or_async(question_text, files)
        result = await coro_or_future
    except Exception as e:
        # This is where the error is likely happening
        return JSONResponse(status_code=500, content={"error": f"Agent error: {str(e)}"})

    # Defensive check: ensure result is a dictionary before using .get()
    output_raw = result.get("output", result) if isinstance(result, dict) else result

    # Check the type of the parsed output before trying to return it
    if isinstance(output_raw, str):
        parsed = _try_extract_json_from_text(output_raw)
        if parsed is not None:
            # If successfully parsed, return the parsed object
            return JSONResponse(content=parsed, media_type="application/json")
        # If not parsed, return the raw string inside a JSON object
        return JSONResponse(content={"output": output_raw}, media_type="application/json")

    # If output_raw is not a string (e.g., a dictionary or list from the agent)
    try:
        return JSONResponse(content=output_raw, media_type="application/json")
    except TypeError:
        return JSONResponse(content={"output": str(output_raw)}, media_type="application/json")