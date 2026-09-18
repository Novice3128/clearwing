"""Tests for the memory subsystem: SessionStore, EpisodicMemory, SemanticMemory, ContextSummarizer."""

import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from clearwing.data.memory.episodic_memory import Episode, EpisodicMemory
from clearwing.data.memory.semantic_memory import Knowledge, SemanticMemory
from clearwing.data.memory.session_store import SessionInfo, SessionStore
from clearwing.data.memory.summarizer import ContextSummarizer
from clearwing.llm import AIMessage, HumanMessage

# =========================================================================
# SessionStore
# =========================================================================


class TestSessionInfo:
    def test_creation(self):
        s = SessionInfo(
            session_id="abc12345",
            target="10.0.0.1",
            model="claude-sonnet-4-6",
            status="running",
            start_time=datetime.now(),
            langgraph_thread_id="thread123",
        )
        assert s.session_id == "abc12345"
        assert s.flags_found == []
        assert s.cost_usd == 0.0


class TestSessionStore:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = SessionStore()
        self.store.BASE_DIR = Path(self._tmpdir)

    def test_create(self):
        session = self.store.create("10.0.0.1", "claude-sonnet-4-6")
        assert len(session.session_id) == 8
        assert session.target == "10.0.0.1"
        assert session.status == "running"
        assert session.langgraph_thread_id

    def test_save_and_load(self):
        session = self.store.create("10.0.0.1", "claude-sonnet-4-6")
        session.cost_usd = 0.05
        session.flags_found = [{"flag": "flag{test}", "context": "found it", "timestamp": "now"}]
        self.store.save(session)

        loaded = self.store.load(session.session_id)
        assert loaded.session_id == session.session_id
        assert loaded.cost_usd == 0.05
        assert len(loaded.flags_found) == 1
        assert loaded.flags_found[0]["flag"] == "flag{test}"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self.store.load("nonexistent")

    def test_list_sessions(self):
        self.store.create("10.0.0.1", "claude-sonnet-4-6")
        self.store.create("10.0.0.2", "claude-opus-4-6")
        sessions = self.store.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_filter_by_target(self):
        self.store.create("10.0.0.1", "claude-sonnet-4-6")
        self.store.create("10.0.0.2", "claude-opus-4-6")
        sessions = self.store.list_sessions(target="10.0.0.1")
        assert len(sessions) == 1
        assert sessions[0].target == "10.0.0.1"

    def test_get_latest(self):
        self.store.create("10.0.0.1", "claude-sonnet-4-6")
        s2 = self.store.create("10.0.0.1", "claude-sonnet-4-6")
        latest = self.store.get_latest()
        assert latest is not None
        # The latest should be s2 since it was created after s1
        assert latest.session_id == s2.session_id

    def test_get_latest_none(self):
        assert self.store.get_latest() is None

    def test_delete(self):
        session = self.store.create("10.0.0.1", "claude-sonnet-4-6")
        self.store.delete(session.session_id)
        with pytest.raises(FileNotFoundError):
            self.store.load(session.session_id)

    def test_delete_nonexistent_is_noop(self):
        self.store.delete("nonexistent")  # should not raise

    def test_degrades_gracefully_when_home_unwritable(self, monkeypatch, tmp_path):
        """#7: an unwritable CLEARWING_HOME (container HOME=/nonexistent)
        must not make construction raise; the store no-ops instead."""
        import clearwing.core.config as config_mod

        blocked = tmp_path / "blocked-home"
        blocked.write_text("")  # a file where a directory is needed
        monkeypatch.setattr(config_mod, "clearwing_home", lambda: blocked)

        store = SessionStore()  # must not raise
        assert store.available is False
        assert store.unavailable_reason
        assert "not writable" in store.unavailable_reason
        # Mutations are no-ops; create still returns an in-memory session.
        session = store.create("10.0.0.1", "claude-sonnet-4-6")
        assert session.session_id
        store.save(session)
        store.delete(session.session_id)
        # Reads behave like an empty store.
        assert store.list_sessions() == []
        assert store.get_latest() is None
        with pytest.raises(FileNotFoundError):
            store.load(session.session_id)

    def test_datetime_serialization(self):
        session = self.store.create("10.0.0.1", "claude-sonnet-4-6")
        session.end_time = datetime(2025, 6, 15, 12, 30, 0)
        self.store.save(session)
        loaded = self.store.load(session.session_id)
        assert isinstance(loaded.start_time, datetime)
        assert isinstance(loaded.end_time, datetime)
        assert loaded.end_time.year == 2025

    def test_deferred_session_row_gets_resolved_model(self):
        """#28: interactive sessions created with model="" (defer to config)
        must carry the graph's resolved model once it exists."""
        from types import SimpleNamespace

        from clearwing.ui.commands.interactive import _sync_session_model

        session = self.store.create("10.0.0.1", model="")
        graph = SimpleNamespace(llm=SimpleNamespace(model_name="glm-5.3"))
        _sync_session_model(session, graph)
        assert session.model == "glm-5.3"
        self.store.save(session)
        assert self.store.load(session.session_id).model == "glm-5.3"

    def test_sync_session_model_noops(self):
        from types import SimpleNamespace

        from clearwing.ui.commands.interactive import _sync_session_model

        # No session → nothing to do; graph without an llm/model_name or an
        # unchanged model must not fabricate or clobber values.
        _sync_session_model(None, SimpleNamespace(llm=SimpleNamespace(model_name="m")))
        session = self.store.create("10.0.0.1", model="kimi-k2")
        _sync_session_model(session, object())
        assert session.model == "kimi-k2"
        _sync_session_model(
            session, SimpleNamespace(llm=SimpleNamespace(model_name="kimi-k2"))
        )
        assert session.model == "kimi-k2"


class TestTuiSessionWriteBack:
    """Three-lens review (F8): the TUI exit write-back (resolved model +
    status) must survive `app.run()` raising — the graph only exists after
    the TUI's on_mount, so a crash used to skip the write-back entirely."""

    def _run(self, monkeypatch, run_outcome):
        import clearwing.ui.commands.interactive as interactive

        saved = {}

        class _FakeStore:
            def save(self, session):
                saved["session"] = session

        monkeypatch.setattr(interactive, "SessionStore", _FakeStore)

        class _FakeApp:
            def __init__(self, **kwargs):
                self._agent_graph = SimpleNamespace(
                    llm=SimpleNamespace(model_name="glm-5.3")
                )

            def run(self):
                run_outcome()

        monkeypatch.setattr(interactive, "ClearwingApp", _FakeApp)

        session = SimpleNamespace(session_id="sec-tui", status="running", model="")
        cli = SimpleNamespace(console=SimpleNamespace(print=lambda *a, **k: None))
        args = SimpleNamespace(
            target="t",
            model=None,
            model_explicit=False,
            base_url=None,
            api_key=None,
        )
        return interactive, saved, session, cli, args

    def test_normal_exit_writes_back_completed_and_model(self, monkeypatch):
        interactive, saved, session, cli, args = self._run(monkeypatch, lambda: None)

        interactive._run_tui(cli, args, session)

        assert session.model == "glm-5.3"
        assert session.status == "completed"
        assert saved["session"] is session

    def test_crashing_tui_still_writes_back_model(self, monkeypatch):
        def boom():
            raise RuntimeError("tui exploded")

        interactive, saved, session, cli, args = self._run(monkeypatch, boom)

        with pytest.raises(RuntimeError):
            interactive._run_tui(cli, args, session)

        # The crash must not lose the write-back: model resolved, row saved.
        assert session.model == "glm-5.3"
        assert session.status == "error"
        assert saved["session"] is session


# =========================================================================
# EpisodicMemory
# =========================================================================


class TestEpisodicMemory:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self._tmpdir) / "test_memory.db")
        self.mem = EpisodicMemory(db_path=self.db_path, session_id="test-session")

    def test_record_and_recall(self):
        self.mem.record("10.0.0.1", "port_found", "Port 22 open (SSH)")
        self.mem.record("10.0.0.1", "port_found", "Port 80 open (HTTP)")
        episodes = self.mem.recall("10.0.0.1")
        assert len(episodes) == 2
        assert all(isinstance(e, Episode) for e in episodes)

    def test_recall_empty(self):
        episodes = self.mem.recall("192.168.1.1")
        assert episodes == []

    def test_recall_limit(self):
        for i in range(10):
            self.mem.record("10.0.0.1", "port_found", f"Port {i}")
        episodes = self.mem.recall("10.0.0.1", limit=3)
        assert len(episodes) == 3

    def test_recall_by_type(self):
        self.mem.record("10.0.0.1", "port_found", "Port 22")
        self.mem.record("10.0.0.1", "vuln_found", "CVE-2021-44228")
        self.mem.record("10.0.0.1", "port_found", "Port 80")

        ports = self.mem.recall_by_type("10.0.0.1", "port_found")
        assert len(ports) == 2
        vulns = self.mem.recall_by_type("10.0.0.1", "vuln_found")
        assert len(vulns) == 1

    def test_search_fts(self):
        self.mem.record("10.0.0.1", "vuln_found", "SQL injection vulnerability in login form")
        self.mem.record("10.0.0.1", "port_found", "Port 3306 MySQL open")
        results = self.mem.search("10.0.0.1", "SQL injection")
        assert len(results) >= 1
        assert "SQL injection" in results[0].content

    def test_metadata_stored(self):
        self.mem.record(
            "10.0.0.1", "port_found", "Port 22", metadata={"protocol": "tcp", "service": "ssh"}
        )
        episodes = self.mem.recall("10.0.0.1")
        assert episodes[0].metadata["service"] == "ssh"

    def test_different_targets_isolated(self):
        self.mem.record("10.0.0.1", "port_found", "Port 22")
        self.mem.record("10.0.0.2", "port_found", "Port 80")
        assert len(self.mem.recall("10.0.0.1")) == 1
        assert len(self.mem.recall("10.0.0.2")) == 1

    def test_session_id_default(self):
        ep = self.mem.record("10.0.0.1", "note_added", "test note")
        assert ep.session_id == "test-session"

    def test_session_id_override(self):
        ep = self.mem.record("10.0.0.1", "note_added", "test", session_id="other")
        assert ep.session_id == "other"


# =========================================================================
# SemanticMemory
# =========================================================================


class TestSemanticMemory:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self._tmpdir) / "test_semantic.db")
        # Force FTS5 fallback by clearing the model cache
        SemanticMemory._model = None
        SemanticMemory._model_checked = False
        self.mem = SemanticMemory(db_path=self.db_path)

    def test_store_and_search(self):
        self.mem.store("successful_exploits", "Used SQL injection on port 3306 to dump database")
        self.mem.store("successful_exploits", "Exploited XSS to steal session cookies")
        results = self.mem.search("SQL injection")
        assert len(results) >= 1

    def test_store_returns_knowledge(self):
        k = self.mem.store("custom_techniques", "Custom technique for bypassing WAF")
        assert isinstance(k, Knowledge)
        assert k.category == "custom_techniques"
        assert k.id > 0

    def test_search_empty(self):
        results = self.mem.search("nonexistent query")
        assert results == []

    def test_search_with_category_filter(self):
        self.mem.store("successful_exploits", "SQL injection worked")
        self.mem.store("tool_usage_patterns", "Use sqlmap with --level=5")
        results = self.mem.search("SQL", category="tool_usage_patterns")
        for r in results:
            assert r.category == "tool_usage_patterns"

    def test_metadata_stored(self):
        self.mem.store("target_profiles", "Target runs Apache", metadata={"target": "10.0.0.1"})
        results = self.mem.search("Apache")
        assert len(results) >= 1
        assert results[0].metadata.get("target") == "10.0.0.1"

    def test_fts_fallback_works(self):
        """Verify FTS5 search works even without sentence-transformers."""
        SemanticMemory._model = None
        SemanticMemory._model_checked = True  # Skip model loading
        mem = SemanticMemory(db_path=self.db_path)
        mem.store("custom_techniques", "buffer overflow exploitation technique")
        results = mem._fts_search("buffer overflow", None, 5)
        assert len(results) >= 1


# =========================================================================
# ContextSummarizer
# =========================================================================


class TestContextSummarizer:
    def setup_method(self):
        self.summarizer = ContextSummarizer()

    def test_should_summarize_false_when_short(self):
        messages = [HumanMessage(content="hello")]
        assert self.summarizer.should_summarize(messages) is False

    def test_should_summarize_true_when_long(self):
        # 120000 tokens * 4 chars = 480000 chars needed to exceed 80% of 150000
        big_msg = HumanMessage(content="x" * 500000)
        assert self.summarizer.should_summarize([big_msg]) is True

    def test_should_summarize_custom_threshold(self):
        msg = HumanMessage(content="x" * 1000)
        # 1000 chars / 4 = 250 tokens. 80% of 200 = 160. 250 > 160 → True
        assert self.summarizer.should_summarize([msg], max_tokens=200) is True

    @pytest.mark.asyncio
    async def test_summarize_preserves_flags(self):
        messages = []
        for i in range(10):
            messages.append(HumanMessage(content=f"Message {i}"))
            messages.append(AIMessage(content=f"Response {i}"))

        # Add a flag-bearing message in the old section
        messages.insert(2, AIMessage(content="Found flag{test_flag_123}"))

        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(first_text="Summary of findings")

        result = await self.summarizer.summarize(messages, mock_llm)

        # The flag-bearing message must stay verbatim in the compacted view
        flag_found = any(
            "flag{test_flag_123}" in getattr(m, "content", "")
            for m in result["view"]
        )
        assert flag_found, "Flag-bearing message was not preserved"

    @pytest.mark.asyncio
    async def test_summarize_keeps_recent_messages(self):
        messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]

        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(first_text="Summary")

        result = await self.summarizer.summarize(messages, mock_llm)

        # Recent 30% (3 messages) should be preserved as-is
        assert any("Msg 9" in getattr(m, "content", "") for m in result["view"])
        assert any("Msg 8" in getattr(m, "content", "") for m in result["view"])
        # The compacted old coverable messages leave the view
        assert not any("Msg 0" in getattr(m, "content", "") for m in result["view"])

    @pytest.mark.asyncio
    async def test_summarize_empty_returns_empty(self):
        mock_llm = AsyncMock()
        result = await self.summarizer.summarize([], mock_llm)
        assert result["view"] == []
        mock_llm.aask_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_reuses_prior_when_nothing_newly_coverable(self):
        # Issue #38: when the old segment holds nothing newly coverable
        # (here: all tool traffic), the LLM must NOT be re-called and the
        # prior summary is returned verbatim.
        from clearwing.llm import ToolMessage

        messages = [
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
            ToolMessage(content="result", tool_call_id="c1"),
        ] * 10

        mock_llm = AsyncMock()
        prior = {"text": "previous summary", "covered_count": 5}

        result = await self.summarizer.summarize(messages, mock_llm, prior=prior)

        mock_llm.aask_text.assert_not_called()
        assert result["text"] == "previous summary"
        assert result["covered_count"] == 5
        assert len(result["view"]) == len(messages)

    @pytest.mark.asyncio
    async def test_summarize_accumulates_prior_text(self):
        messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]
        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(first_text="new summary")

        prior = {"text": "previous summary", "covered_count": 3}
        result = await self.summarizer.summarize(messages, mock_llm, prior=prior)

        user_payload = mock_llm.aask_text.call_args.kwargs["user"]
        assert "previous summary" in user_payload
        assert result["covered_count"] == 3 + 7  # prior + newly covered (10 * 0.7)

    @pytest.mark.asyncio
    async def test_summarize_passes_prompt_cache_params(self):
        # Issue #38: the summarization call itself must carry the cache
        # hint, mirroring the main agent loop.
        messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]
        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(first_text="summary")

        await self.summarizer.summarize(
            messages, mock_llm, prompt_cache_key="sess-1:wi-1"
        )

        kwargs = mock_llm.aask_text.call_args.kwargs
        assert kwargs["cache_prefix"] is True
        assert kwargs["prompt_cache_key"] == "sess-1:wi-1"

    @pytest.mark.asyncio
    async def test_summarize_returns_usage_when_provider_reported_it(self):
        # The summary LLM call is real spend: its usage must ride on the
        # result so callers (runtime, hunter) can book it.
        messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]

        class _Usage:
            prompt_tokens = 700
            completion_tokens = 90
            total_tokens = 790

            class prompt_tokens_details:  # noqa: N805 - simple namespace
                cached_tokens = 100

        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(
            first_text="summary", usage=_Usage()
        )

        result = await self.summarizer.summarize(messages, mock_llm)

        assert result["usage"] == {
            "input_tokens": 700,
            "output_tokens": 90,
            "cached_tokens": 100,
        }

    @pytest.mark.asyncio
    async def test_summarize_usage_none_without_concrete_tokens(self):
        # Fakes/older builds without real usage figures must yield None —
        # never a hallucinated number.
        messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]
        mock_llm = AsyncMock()
        # MagicMock auto-attributes are not ints.
        mock_llm.aask_text.return_value = MagicMock(first_text="summary")

        result = await self.summarizer.summarize(messages, mock_llm)
        assert result["usage"] is None

    @pytest.mark.asyncio
    async def test_summarize_usage_none_on_early_returns(self):
        # Empty history and nothing-newly-coverable skip the LLM entirely.
        mock_llm = AsyncMock()
        result = await self.summarizer.summarize([], mock_llm)
        assert result["usage"] is None

        from clearwing.llm import ToolMessage

        messages = [
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
            ToolMessage(content="result", tool_call_id="c1"),
        ] * 10
        result = await self.summarizer.summarize(messages, mock_llm, prior=None)
        assert result["usage"] is None
        mock_llm.aask_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_returns_served_model_echo(self):
        # Codex PR-63 r2: the provider's model echo rides on the result so
        # callers (runtime, hunter) can AUDIT the summary call under the
        # model that actually served it. Additive — the #38 four-key
        # contract is untouched; no exact-key-set assertion on purpose.
        messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]

        class _Usage:
            prompt_tokens = 12
            completion_tokens = 6
            total_tokens = 18
            prompt_tokens_details = None

        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(
            first_text="summary", usage=_Usage(), provider_model_name="echoed-model-7"
        )

        result = await self.summarizer.summarize(messages, mock_llm)

        assert result["served_model"] == "echoed-model-7"

        # A response WITHOUT the echo attribute (test doubles, older
        # builds) must yield None, never a MagicMock auto-attribute.
        mock_llm.aask_text.return_value = MagicMock(first_text="summary")
        result = await self.summarizer.summarize(messages, mock_llm)
        assert result["served_model"] is None

    @pytest.mark.asyncio
    async def test_summarize_served_model_none_on_early_returns(self):
        # Paths that never reach an LLM response carry the key as None.
        mock_llm = AsyncMock()
        empty = await self.summarizer.summarize([], mock_llm)
        assert empty["served_model"] is None

        from clearwing.llm import ToolMessage

        tool_traffic = [
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
            ToolMessage(content="result", tool_call_id="c1"),
        ] * 10
        reused = await self.summarizer.summarize(
            tool_traffic, mock_llm, prior={"text": "prior", "covered_count": 1}
        )
        assert reused["served_model"] is None
        mock_llm.aask_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_summary_does_not_compact_history(self):
        # An empty/whitespace summary must not replace the covered messages
        # with nothing: the prior state and the original view survive, and
        # the billed usage is still surfaced.
        messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]
        prior = {"text": "previous summary", "covered_count": 4}

        class _Usage:
            prompt_tokens = 500
            completion_tokens = 0
            total_tokens = 500

        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(
            first_text="   ", usage=_Usage()
        )  # whitespace-only summary

        result = await self.summarizer.summarize(messages, mock_llm, prior=prior)

        assert result["text"] == "previous summary"
        assert result["covered_count"] == 4
        assert list(result["view"]) == messages  # history untouched
        # The call still happened and cost money — usage is returned.
        assert result["usage"] == {
            "input_tokens": 500,
            "output_tokens": 0,
            "cached_tokens": 0,
        }

    def test_dict_tool_messages_are_not_coverable(self):
        # Dict-shaped (legacy LangChain-style) tool traffic must stay
        # verbatim: covering a tool result (or an assistant turn carrying
        # tool_calls) orphans the paired tool_use — provider 400.
        tool_result = {"role": "tool", "tool_call_id": "c1", "content": "result"}
        assistant_with_calls = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "t", "args": {}}],
        }
        plain_user = {"role": "user", "content": "hello"}

        assert not self.summarizer._is_coverable(tool_result)
        assert not self.summarizer._is_coverable(assistant_with_calls)
        # Plain dict messages remain summarizable as before.
        assert self.summarizer._is_coverable(plain_user)

    @pytest.mark.asyncio
    async def test_dict_tool_messages_survive_compaction(self):
        # End to end: dict tool traffic in the old segment stays in the
        # compacted view.
        messages: list = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "t", "args": {}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "tool output"},
        ] * 5
        messages.extend(HumanMessage(content=f"Msg {i}") for i in range(10))

        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(first_text="summary")

        result = await self.summarizer.summarize(messages, mock_llm)

        view_ids = {id(m) for m in result["view"]}
        for m in messages[:10]:  # the old segment's dict tool traffic
            assert id(m) in view_ids

    @pytest.mark.asyncio
    async def test_dict_user_content_reaches_summary_input(self):
        # PR #44 review P1: dict-shaped user inputs used to be selected as
        # coverable but dropped from the summary LLM's text_block
        # (``getattr`` on a dict yields None) — then removed from the
        # committed view, permanently losing the early user goals. The
        # covered content must appear in the summary input.
        goal = {"role": "user", "content": "Find the hidden flag on the target"}
        messages: list = [goal]
        messages.extend(
            {"role": "assistant", "content": f"assistant note {i}"} for i in range(9)
        )

        mock_llm = AsyncMock()
        mock_llm.aask_text.return_value = MagicMock(first_text="summary")

        result = await self.summarizer.summarize(messages, mock_llm)

        user_payload = mock_llm.aask_text.call_args.kwargs["user"]
        assert "[user]: Find the hidden flag on the target" in user_payload
        assert "[assistant]: assistant note 0" in user_payload
        # And the covered dict messages did leave the committed view.
        assert not any(m is goal for m in result["view"])

    def test_dict_flag_message_is_not_coverable(self):
        # Flag detection must read dict content too: a dict message bearing
        # a flag stays verbatim instead of being summarized away.
        flagged = {"role": "user", "content": "captured flag{dict_flag_1}"}
        assert not self.summarizer._is_coverable(flagged)

    def test_estimate_tokens_reads_dict_content(self):
        messages = [{"role": "user", "content": "x" * 400}]
        # 400 chars / 4 = 100 tokens; dict content must actually be read.
        assert self.summarizer._estimate_tokens(messages) == 100
