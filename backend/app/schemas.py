from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=5000)
    model_id: Literal["starter", "ai-lead"] = "ai-lead"


class BikeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: Literal["starter", "ai-lead"] | None = None
    season: int = Field(ge=1, le=4)
    year: int = Field(ge=0, le=1, description="UCI dataset encoding: 0=2011, 1=2012")
    month: int = Field(ge=1, le=12)
    hour: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weather: int = Field(ge=1, le=4, alias="weathersit")
    temperature_c: float = Field(ge=0, le=41)
    feels_like_c: float = Field(ge=0, le=50)
    humidity_percent: float = Field(ge=0, le=100)
    wind_speed_kmh: float = Field(ge=0, le=67)

    def dataset_features(self) -> dict[str, float | int]:
        return {
            "season": self.season,
            "yr": self.year,
            "mnth": self.month,
            "hr": self.hour,
            "holiday": self.holiday,
            "weekday": self.weekday,
            "workingday": self.workingday,
            "weathersit": self.weather,
            "temp": self.temperature_c / 41.0,
            "atemp": self.feels_like_c / 50.0,
            "hum": self.humidity_percent / 100.0,
            "windspeed": self.wind_speed_kmh / 67.0,
        }
