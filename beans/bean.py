from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ResponseData(BaseModel):
    """Response data model with timestamp, code, and MAC address"""
    time: str
    code: str
    mac: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "time": "2025-10-16T18:30:00+05:30",
                "code": "SUCCESS",
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
    drowsy: Optional[bool] = None
    yawning: Optional[bool] = None
    microsleep: Optional[bool] = None
    perclos: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "time": "2025-10-16T18:30:00+05:30",
                "code": "BEHAVIOR_EVENT",
                "mac": "48:89:E7:FA:15:AE",
                "drowsy": True,
                "yawning": False,
                "microsleep": False,
                "perclos": 0.75
            }
        }