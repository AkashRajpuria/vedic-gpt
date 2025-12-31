from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class WordByWordItem(BaseModel):
    sa: str = Field(..., description="Sanskrit token/word (usually Devanagari)")
    en: Optional[str] = Field(None, description="English gloss for the Sanskrit token")
    hi: Optional[str] = Field(None, description="Hindi gloss for the Sanskrit token")


class VerseRecord(BaseModel):
    source: Literal["bhagavad_gita"] = Field(..., description="Data source identifier")
    chapter: int = Field(..., ge=1)
    verse: int = Field(..., ge=1)

    sanskrit_devanagari: str = Field(..., min_length=1)
    english: str = Field(..., min_length=1)
    hindi: str = Field(..., min_length=1)

    # Optional enrichments
    word_by_word: Optional[list[WordByWordItem]] = None
    notes: Optional[str] = None

    def ref(self) -> str:
        return f"BG {self.chapter}.{self.verse}"


