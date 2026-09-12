"""
Clarity v0.17 — the data layer, and what belongs in it.

Files were fine for fifteen chapters. They stop being fine at the point where two
processes write at once, where a question needs an answer about last Tuesday, and where
somebody asks which tenant saw which document.

Five tables, and the argument for each is a question somebody will ask you:

    documents     what do we hold, for whom, and which version did we answer from
    runs          what did we answer, to whom, at what cost
    steps         what did it do on the way — the trace, joined to the run
    evaluations   how did it score, against which version of the golden set
    feedback      what did the user think, attached to the run they thought it about

The shape matters more than the engine. This file uses SQLite so the book runs
anywhere; the DDL is ordinary SQL and the same models point at Postgres by changing
one URL.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine, func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


def _id() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    # Every row in every table carries this. Chapter 26 enforces it; the schema has
    # to make it possible, and retrofitting a tenant column to a live system is a
    # migration nobody enjoys.
    tenant: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    tenant: Mapped[str] = mapped_column(String(64), index=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now,
                                                    index=True)
    steps: Mapped[list["Step"]] = relationship(back_populates="run",
                                               cascade="all, delete-orphan")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="run")


class Step(Base):
    __tablename__ = "steps"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    n: Mapped[int] = mapped_column(Integer)
    tool: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Arguments, not results. §23.5 is the whole reason for that distinction, and it
    # applies to a database you own as much as to a vendor you do not.
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    result_chars: Mapped[int] = mapped_column(Integer, default=0)
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    failed: Mapped[bool] = mapped_column(Boolean, default=False)
    run: Mapped[Run] = relationship(back_populates="steps")


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    suite_version: Mapped[int] = mapped_column(Integer)
    layer: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    git_sha: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now,
                                                    index=True)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    verdict: Mapped[str] = mapped_column(String(16))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    run: Mapped[Run] = relationship(back_populates="feedback")


def open_store(url: str = "sqlite:///data/meridian/clarity.db"):
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine
