# SPDX-License-Identifier: Apache-2.0
"""Growing conversation workload for ``lmcache bench engine``.

Simulates multi-round conversations where context grows from below
the DPD routing threshold to above it. Demonstrates conditional
DPD behavior and bidirectional NIXL cache probe benefit.
"""

from dataclasses import dataclass, field
import asyncio

from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class GrowingConversationConfig:
    """Config for the growing-conversation workload."""

    num_conversations: int = 60
    num_rounds: int = 10
    system_prompt_length: int = 500
    user_message_length: int = 500
    output_length: int = 200
    qps: float = 4.0

    @classmethod
    def resolve(cls, **kwargs) -> "GrowingConversationConfig":
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


@dataclass
class _Conversation:
    conv_id: int
    system_prompt: str
    exchanges: list = field(default_factory=list)
    in_flight: bool = False

    def build_messages(self, query: str) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        for q, a in self.exchanges:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": query})
        return msgs


class GrowingConversationWorkload(BaseWorkload):
    """Workload where conversations grow past the DPD routing threshold.

    Each conversation starts with a short system prompt (~500 tokens)
    and grows by ~500 tokens per round. With a 4096-token threshold,
    rounds 1-6 stay below threshold (handled locally), and rounds 7+
    cross the threshold (routed to prefiller in DPD mode).
    """

    def __init__(self, config, request_sender, stats_collector,
                 progress_monitor, seed=42):
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._conversations = self._create_conversations()
        self._round_index = 0
        self._conv_index = 0
        self._interval = 1.0 / config.qps
        self._global_index = 0
        self._pending_tasks: set[asyncio.Task] = set()

    def log_config(self) -> None:
        c = self._config
        B, C, Y, R = "\033[1m", "\033[96m", "\033[93m", "\033[0m"
        threshold_round = max(1, 4096 // (c.system_prompt_length + c.user_message_length + c.output_length))
        print(
            f"{B}{'═'*50}{R}\n"
            f"{B} Workload: {C}growing-conversation{R}\n"
            f"{B}{'─'*50}{R}\n"
            f"  Conversations:    {Y}{c.num_conversations}{R}\n"
            f"  Rounds:           {Y}{c.num_rounds}{R}\n"
            f"  System prompt:    {Y}{c.system_prompt_length}{R} tokens\n"
            f"  User msg length:  {Y}{c.user_message_length}{R} tokens\n"
            f"  Output length:    {Y}{c.output_length}{R} tokens\n"
            f"  QPS:              {Y}{c.qps}{R}\n"
            f"  ~Threshold round: {Y}{threshold_round}{R} (at 4096 tokens)\n"
            f"{B}{'═'*50}{R}"
        )

    def _create_conversations(self):
        convs = []
        for i in range(self._config.num_conversations):
            sp = f"Conversation {i}. You are a helpful assistant. " + " ".join(
                ["help"] * self._config.system_prompt_length
            )
            convs.append(_Conversation(i, sp))
        return convs

    def _make_user_message(self, conv_id, round_idx):
        return f"Round {round_idx} question for conversation {conv_id}. " + " ".join(
            ["tell"] * self._config.user_message_length
        )

    async def warmup(self) -> None:
        """Send one warmup request per conversation with max_tokens=2.

        Uses max_tokens=2 to ensure compatibility with DPD routers
        that consume the first token during prefill-decode handoff.
        """
        num = len(self._conversations)
        for i, conv in enumerate(self._conversations):
            rid = f"warmup_c{conv.conv_id}"
            query = f"Hello, conversation {conv.conv_id}."
            msgs = conv.build_messages(query)
            self._progress_monitor.log_message(f"Warmup {i + 1}/{num}")
            self._progress_monitor.on_request_sent(rid)
            result = await self._request_sender.send_warmup_request(
                rid, msgs, max_tokens=2, session_id=f"session-{conv.conv_id}",
            )
            if not result.successful:
                self._progress_monitor.log_message(
                    f"Warmup conversation {conv.conv_id} failed: {result.error}"
                )
        self._progress_monitor.log_message(f"Warmup complete: {num} conversations")

    async def step(self, time_offset: float) -> float:
        if self._round_index >= self._config.num_rounds:
            if self._pending_tasks:
                await asyncio.wait(self._pending_tasks,
                                   return_when=asyncio.FIRST_COMPLETED)
                return 0.0
            return -1.0

        conv = self._conversations[self._conv_index]
        if conv.in_flight:
            return time_offset + 0.01

        conv.in_flight = True
        query = self._make_user_message(conv.conv_id, self._round_index)
        rid = f"c{conv.conv_id}_r{self._round_index}"
        msgs = conv.build_messages(query)

        self._progress_monitor.on_request_sent(rid)
        task = asyncio.create_task(self._send(rid, msgs, conv, query))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

        self._conv_index += 1
        if self._conv_index >= len(self._conversations):
            self._conv_index = 0
            self._round_index += 1
            self._progress_monitor.log_message(
                f"Starting round {self._round_index}/{self._config.num_rounds}"
            )

        self._global_index += 1
        return self._global_index * self._interval

    async def _send(self, rid, msgs, conv, query):
        sid = f"session-{conv.conv_id}"
        result = await self._request_sender.send_request(
            rid, msgs, max_tokens=self._config.output_length,
            session_id=sid,
        )
        if result.successful:
            conv.exchanges.append((query, "response"))
        conv.in_flight = False

    def on_request_finished(self, request_id: str, output: str) -> None:
        pass
