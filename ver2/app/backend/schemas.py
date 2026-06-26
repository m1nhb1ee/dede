from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    url: str | None = Field(default=None, description="Public post URL to scrape best-effort.")
    title: str = Field(default="", description="Post title/caption.")
    body: str = Field(default="", description="Post body/content.")
    upvotes: float = Field(default=0.0, ge=0.0)
    num_comments: float = Field(default=0.0, ge=0.0)
    created_utc: int | None = Field(default=None, description="Unix timestamp. Defaults to now.")
    translate: bool = Field(default=True, description="Translate extracted text to English.")

    @model_validator(mode="after")
    def require_url_or_text(self):
        if not self.url and not (self.title.strip() or self.body.strip()):
            raise ValueError("Provide either url or title/body text.")
        return self


class PredictResponse(BaseModel):
    p_text_stage1: float
    p_final_depression_risk: float
    predicted_label_at_0_5: int
    title_en_clean: str
    body_en_clean: str
    upvotes: float
    num_comments: float
    created_utc: int
    source_url: str | None
    note: str


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
