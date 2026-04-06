# SPDX-License-Identifier: Apache-2.0
"""Mixed workload for ``lmcache bench engine``.

Combines multi-round chat sessions (shared prefix) with unique
random prompts (no shared prefix) at a configurable ratio.
"""

from dataclasses import dataclass, field
import asyncio
import random

from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class MixedWorkloadConfig:
    """Config for the mixed workload."""

    shared_prompt_length: int = 2000
    chat_history_length: int = 10000
    unique_prompt_length: int = 12000
    user_input_length: int = 50
    output_length: int = 200
    qps: float = 6.0
    duration: float = 180.0
    num_chat_sessions: int = 25
    unique_ratio: float = 0.5

    @classmethod
    def resolve(cls, **kwargs) -> "MixedWorkloadConfig":
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


@dataclass
class _Session:
    session_id: int
    system_prompt: str
    history_text: str
    exchanges: list = field(default_factory=list)
    in_flight: bool = False

    def build_messages(self, query: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self.history_text:
            msgs.append({"role": "user", "content": self.history_text})
            msgs.append({"role": "assistant", "content": "Understood."})
        for q, a in self.exchanges:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": query})
        return msgs


class MixedWorkload(BaseWorkload):
    """Workload mixing multi-round chat with unique random prompts."""

    def __init__(self, config, request_sender, stats_collector,
                 progress_monitor, seed=42):
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._rng = random.Random(seed)
        self._sessions = self._create_sessions()
        self._global_index = 0
        self._unique_index = 0
        self._interval = 1.0 / config.qps
        self._pending_tasks: set[asyncio.Task] = set()

    def log_config(self) -> None:
        c = self._config
        B, C, Y, R = "\033[1m", "\033[96m", "\033[93m", "\033[0m"
        print(
            f"{B}{'═'*50}{R}\n"
            f"{B} Workload: {C}mixed{R}\n"
            f"{B}{'─'*50}{R}\n"
            f"  Chat sessions:    {Y}{c.num_chat_sessions}{R}\n"
            f"  Unique ratio:     {Y}{c.unique_ratio:.0%}{R}\n"
            f"  Prompt length:    {Y}{c.shared_prompt_length}{R} tokens\n"
            f"  History length:   {Y}{c.chat_history_length}{R} tokens\n"
            f"  Unique length:    {Y}{c.unique_prompt_length}{R} tokens\n"
            f"  Output length:    {Y}{c.output_length}{R} tokens\n"
            f"  QPS:              {Y}{c.qps}{R}\n"
            f"  Duration:         {Y}{c.duration}s{R}\n"
            f"{B}{'═'*50}{R}"
        )

    def _create_sessions(self):
        sessions = []
        for i in range(self._config.num_chat_sessions):
            sp = f"Session {i}. You are helpful. " + " ".join(
                ["help"] * self._config.shared_prompt_length
            )
            hist = f"[Session {i} history] " + " ".join(
                ["hi"] * self._config.chat_history_length
            )
            sessions.append(_Session(i, sp, hist))
        return sessions

    def _make_unique_prompt(self):
        idx = self._unique_index
        self._unique_index += 1
        rng = random.Random(idx * 31337)
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy",
                 "dog", "hello", "world", "data", "science", "machine",
                 "learning", "deep", "neural", "network", "model"]
        body = " ".join(rng.choice(words)
                        for _ in range(self._config.unique_prompt_length))
        return f"Document {idx}: {body}\n\nSummarize briefly."

    async def warmup(self) -> None:
        n = len(self._sessions)
        for s in self._sessions:
            rid = f"warmup_s{s.session_id}"
            msgs = s.build_messages("Hello")
            self._progress_monitor.log_message(f"Warmup {s.session_id+1}/{n}")
            self._progress_monitor.on_request_sent(rid)
            await self._request_sender.send_warmup_request(
                rid, msgs, max_tokens=2,
                session_id=f"session-{s.session_id}",
            )
        self._progress_monitor.log_message(f"Warmup complete: {n} sessions")

    async def step(self, time_offset: float) -> float:
        if time_offset >= self._config.duration:
            if self._pending_tasks:
                await asyncio.wait(self._pending_tasks,
                                   return_when=asyncio.FIRST_COMPLETED)
                return 0.0
            return -1.0

        is_unique = self._rng.random() < self._config.unique_ratio

        if is_unique:
            prompt = self._make_unique_prompt()
            rid = f"unique_{self._global_index}"
            msgs = [{"role": "user", "content": prompt}]
            task = asyncio.create_task(self._send(rid, msgs))
        else:
            target = self._global_index % len(self._sessions)
            s = self._sessions[target]
            if s.in_flight:
                return time_offset + 0.01
            s.in_flight = True
            query = " ".join(["tell"] * self._config.user_input_length)
            rid = f"chat_{self._global_index}"
            msgs = s.build_messages(query)
            task = asyncio.create_task(self._send(
                rid, msgs, session=s, query=query,
            ))

        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        self._global_index += 1
        return self._global_index * self._interval

    async def _send(self, rid, msgs, session=None, query=None):
        sid = f"session-{session.session_id}" if session else None
        self._progress_monitor.on_request_sent(rid)
        result = await self._request_sender.send_request(
            rid, msgs, max_tokens=self._config.output_length,
            session_id=sid,
        )
        if session:
            if result.successful:
                session.exchanges.append((query, ""))
            session.in_flight = False

    def on_request_finished(self, request_id: str, output: str) -> None:
        pass
