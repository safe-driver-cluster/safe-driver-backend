from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ResponseData(BaseModel):
    """Response data model with timestamp, code, and MAC address"""
    time: str
    respose_code: str
    response_msg: str
    mac: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "time": "2025-10-16T18:30:00+05:30",
                "respose_code": "00",
                "response_msg": "OK",
                "mac": "48:89:E7:FA:15:AE"
            }
        }


class ApiResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Any] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {
                    "time": "2025-10-16T18:30:00+05:30",
                    "code": "SUCCESS",
                    "mac": "48:89:E7:FA:15:AE"
                }
            }
        }


class BehaviorResponseData(ResponseData):
    """Extended response data for behavior events"""
    tag: str = None
    type: str = None
    message: str = None
    time: str = None
    data: Any = None

class CommonResponse(BaseModel):
    """Common response model for generic API responses"""
    success: bool
    message: str
    data: Optional[Any] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Device registered successfully",
                "data": {
                    "time": "2025-10-16T18:30:00+05:30",
                    "code": "DEVICE_REGISTERED",
                    "mac": "48:89:E7:FA:15:AE"
                }
            }
        }