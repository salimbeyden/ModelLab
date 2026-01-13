"""
Exception Handlers: Centralized error handling for the API.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Callable, TypeVar, Any
import functools
import traceback

T = TypeVar("T")


class APIError(Exception):
    """Base API exception with status code."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(APIError):
    """Resource not found (404)."""
    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found" if resource_id is None else f"{resource} {resource_id} not found"
        super().__init__(message, status_code=404)


class ValidationError(APIError):
    """Validation error (400)."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class ConflictError(APIError):
    """Resource conflict (409)."""
    def __init__(self, message: str):
        super().__init__(message, status_code=409)


def handle_service_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle common service exceptions and convert to HTTP exceptions.
    Use on route handlers to reduce boilerplate.
    """
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e) or "Resource not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e) or "Resource not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
