"""API schemas (spec §4c).

Pydantic v2 models. Field names match the spec's payload sketches exactly so
the UI layer and any external consumer can be built against the spec, not
against implementation details.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Input primitives ────────────────────────────────────────────────────────
class Node(BaseModel):
    id: int
    lat: Optional[float] = None
    lng: Optional[float] = None
    demand: float = 0.0
    tw_early: Optional[float] = None
    tw_late: Optional[float] = None
    service_time: float = 0.0


class Edge(BaseModel):
    i: int
    j: int
    dist: float
    tau0: float
    cap: float = 1.0e6
    emis: Optional[float] = None
    speed0: Optional[float] = None


class Network(BaseModel):
    nodes: List[Node]
    edges: Optional[List[Edge]] = None
    osm_place: Optional[str] = None

    @field_validator("nodes")
    @classmethod
    def _at_least_one_node(cls, v: List[Node]) -> List[Node]:
        if not v:
            raise ValueError("network.nodes must be non-empty")
        return v


class Vehicle(BaseModel):
    vehicle_id: int
    capacity: float


class Weights(BaseModel):
    w1: float = 0.3
    w2: float = 0.5
    w3: float = 0.2
    w4: float = 0.0

    @field_validator("w1", "w2", "w3", "w4")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weights must be ≥ 0")
        return v

    def sum123(self) -> float:
        return self.w1 + self.w2 + self.w3


class EventShockIn(BaseModel):
    name: str
    edges: List[tuple[int, int]]
    start: float
    end: float
    extra_vol: float


class SolverConfig(BaseModel):
    M: int = 32
    T: int = 300
    seed: int = 42
    use_gpu: bool = False
    gls: bool = True
    diversity: bool = True
    time_dependent: bool = False
    events: List[EventShockIn] = Field(default_factory=list)


class SolveRequest(BaseModel):
    network: Network
    fleet: List[Vehicle]
    weights: Weights = Weights()
    solver: SolverConfig = SolverConfig()

    @field_validator("fleet")
    @classmethod
    def _at_least_one_vehicle(cls, v: List[Vehicle]) -> List[Vehicle]:
        if not v:
            raise ValueError("fleet must contain at least one vehicle")
        return v


class TrafficUpdate(BaseModel):
    edges: List[dict]     # [{i, j, new_vol}]
    at_time: float


# ── Output primitives ───────────────────────────────────────────────────────
class RouteOut(BaseModel):
    vehicle_id: int
    sequence: List[int]
    arrival_times: List[float]
    load: float
    distance: float
    time: float
    emissions: float
    reasoning: str = ""


class LateViolation(BaseModel):
    customer: int
    minutes_late: float
    penalty: float


class OverloadViolation(BaseModel):
    vehicle: int
    amount: float
    penalty: float


class ViolationsOut(BaseModel):
    late: List[LateViolation] = Field(default_factory=list)
    overload: List[OverloadViolation] = Field(default_factory=list)


class TotalsOut(BaseModel):
    distance: float
    time: float
    emissions: float
    cost_Z: float
    cost_Z_report: float
    fleet_utilization: float


class SolveResponse(BaseModel):
    routes: List[RouteOut]
    total: TotalsOut
    violations: ViolationsOut
    convergence: List[float]
    runtime_s: float
    feasible: bool
    job_id: Optional[str] = None


class ComparisonDelta(BaseModel):
    distance: float
    time: float
    emissions: float
    late_deliveries: int


class ComparisonResponse(BaseModel):
    a: SolveResponse
    b: SolveResponse
    delta: ComparisonDelta


class JobStatus(BaseModel):
    job_id: str
    state: Literal["pending", "running", "done", "error"]
    best_cost: Optional[float] = None
    iteration: Optional[int] = None
    eta_s: Optional[float] = None
    error: Optional[str] = None


class KPIReport(BaseModel):
    job_id: str
    total: TotalsOut
    violations: ViolationsOut
    routes: List[RouteOut]
    convergence: List[float]


# ── Common error envelope ───────────────────────────────────────────────────
class ErrorEnvelope(BaseModel):
    error: str
    detail: str
    model_config = ConfigDict(extra="forbid")
