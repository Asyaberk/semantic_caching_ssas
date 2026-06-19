"""Curated, low-risk demo questions with known cube and MDX mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    title: str
    question: str
    cube_name: str
    mdx: str
    category: str
    description: str


GOLDEN_QUESTIONS: tuple[GoldenQuestion, ...] = (
    GoldenQuestion(
        id="waiting-count-total",
        title="Total waiting count",
        question="What is the total waiting count?",
        cube_name="cubeWaiting",
        mdx="SELECT {[Measures].[Waitings Count]} ON COLUMNS FROM [cubeWaiting]",
        category="Aggregates",
        description="Safest first check for the Waiting cube.",
    ),
    GoldenQuestion(
        id="waiting-time-total",
        title="Total waiting time",
        question="What is the total waiting time?",
        cube_name="cubeWaiting",
        mdx="SELECT {[Measures].[WaitingTime]} ON COLUMNS FROM [cubeWaiting]",
        category="Aggregates",
        description="Checks the main waiting duration measure.",
    ),
    GoldenQuestion(
        id="vessel-order-count-total",
        title="Total vessel orders",
        question="What is the total vessel order count?",
        cube_name="cubeVesselOrder",
        mdx="SELECT {[Measures].[Vessel Order Count]} ON COLUMNS FROM [cubeVesselOrder]",
        category="Aggregates",
        description="Safest first check for vessel order volume.",
    ),
    GoldenQuestion(
        id="vessel-operation-time-total",
        title="Total vessel operation time",
        question="What is the total vessel operation time?",
        cube_name="cubeVesselOrder",
        mdx="SELECT {[Measures].[OperationTime]} ON COLUMNS FROM [cubeVesselOrder]",
        category="Operations",
        description="Checks operation time without member filters.",
    ),
    GoldenQuestion(
        id="accruement-count-total",
        title="Total accruement count",
        question="What is the total accruement count?",
        cube_name="cubeAccruement",
        mdx="SELECT {[Measures].[Accruement Count]} ON COLUMNS FROM [cubeAccruement]",
        category="Finance",
        description="Safest first check for the Accruement cube.",
    ),
    GoldenQuestion(
        id="credit-debit-count-total",
        title="Total credit/debit count",
        question="What is the total credit debit count?",
        cube_name="cubeCreditDebit",
        mdx="SELECT {[Measures].[Credit Debit Count]} ON COLUMNS FROM [cubeCreditDebit]",
        category="Finance",
        description="Safest first check for the CreditDebit cube.",
    ),
)


def list_golden_questions() -> list[dict]:
    return [item.__dict__.copy() for item in GOLDEN_QUESTIONS]
