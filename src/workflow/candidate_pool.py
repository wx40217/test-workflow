"""候选测试用例池，用于多 Agent 质量图共享状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    """单个候选及其质量证据。"""

    id: str
    content: str
    source_agent: str
    round: int
    review_summary: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    is_valid: bool = False
    created_at_step: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source_agent": self.source_agent,
            "round": self.round,
            "review_summary": dict(self.review_summary),
            "validation_summary": dict(self.validation_summary),
            "quality_score": self.quality_score,
            "is_valid": self.is_valid,
            "created_at_step": self.created_at_step,
        }


class CandidatePool:
    """管理候选添加、评分更新、最佳候选选择和池大小裁剪。"""

    def __init__(self, records: list[dict[str, Any]] | None = None, *, max_size: int = 5):
        self.max_size = max(1, max_size)
        self._records: list[CandidateRecord] = []
        for record in records or []:
            self._records.append(CandidateRecord(**record))

    def add_candidate(
        self,
        content: str,
        *,
        source_agent: str,
        round: int,
        created_at_step: str,
    ) -> CandidateRecord:
        record = CandidateRecord(
            id=f"c{self._next_id_number()}",
            content=content or "",
            source_agent=source_agent,
            round=round,
            created_at_step=created_at_step,
        )
        self._records.append(record)
        self._trim(protected_id=record.id)
        return record

    def update_quality(
        self,
        candidate_id: str,
        *,
        review_summary: dict[str, Any] | None = None,
        validation_summary: dict[str, Any] | None = None,
        quality_score: float | None = None,
        is_valid: bool | None = None,
    ) -> CandidateRecord | None:
        record = self.get(candidate_id)
        if record is None:
            return None
        if review_summary is not None:
            record.review_summary = dict(review_summary)
        if validation_summary is not None:
            record.validation_summary = dict(validation_summary)
        if quality_score is not None:
            record.quality_score = float(quality_score)
        if is_valid is not None:
            record.is_valid = bool(is_valid)
        return record

    def get(self, candidate_id: str) -> CandidateRecord | None:
        for record in self._records:
            if record.id == candidate_id:
                return record
        return None

    def best(self) -> CandidateRecord | None:
        if not self._records:
            return None
        return max(
            self._records,
            key=lambda item: (
                item.is_valid,
                item.quality_score,
                len(item.content.strip()),
                -item.round,
            ),
        )

    def scores(self) -> list[float]:
        return [record.quality_score for record in self._records]

    def to_list(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def _trim(self, protected_id: str = "") -> None:
        if len(self._records) <= self.max_size:
            return
        best = self.best()
        kept: list[CandidateRecord] = []
        protected = self.get(protected_id) if protected_id else None
        if protected is not None:
            kept.append(protected)
        if best is not None:
            self._append_unique(kept, best)
        for record in reversed(self._records):
            if record.id not in {item.id for item in kept}:
                kept.append(record)
            if len(kept) >= self.max_size:
                break
        self._records = kept[: self.max_size]

    @staticmethod
    def _append_unique(records: list[CandidateRecord], record: CandidateRecord) -> None:
        if record.id not in {item.id for item in records}:
            records.append(record)

    def _next_id_number(self) -> int:
        max_id = 0
        for record in self._records:
            if record.id.startswith("c") and record.id[1:].isdigit():
                max_id = max(max_id, int(record.id[1:]))
        return max_id + 1
