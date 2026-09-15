"""Tests for the CI/CD non-interactive runner, result dataclass, and SARIF generator."""

from __future__ import annotations

import pytest

from clearwing.runners.cicd import CICDResult, CICDRunner, SARIFGenerator


class TestCICDResult:
    """Verify the CICDResult dataclass has the expected fields and defaults."""

    def test_fields_present(self):
        result = CICDResult(
            exit_code=0,
            target="192.168.1.1",
            depth="quick",
            findings=[],
            duration_seconds=12.5,
            cost_usd=0.03,
            tokens_used=1500,
            output_path=None,
        )
        assert result.exit_code == 0
        assert result.target == "192.168.1.1"
        assert result.depth == "quick"
        assert result.findings == []
        assert result.duration_seconds == 12.5
        assert result.cost_usd == 0.03
        assert result.tokens_used == 1500
        assert result.output_path is None

    def test_findings_populated(self):
        findings = [
            {
                "description": "SQL Injection",
                "severity": "critical",
                "cve": "CVE-2024-1234",
                "details": "Found in login form",
            }
        ]
        result = CICDResult(
            exit_code=2,
            target="http://example.com",
            depth="deep",
            findings=findings,
            duration_seconds=300.0,
            cost_usd=1.25,
            tokens_used=50000,
            output_path="/tmp/report.json",
        )
        assert result.exit_code == 2
        assert len(result.findings) == 1
        assert result.findings[0]["severity"] == "critical"
        assert result.output_path == "/tmp/report.json"

    def test_output_path_string(self):
        result = CICDResult(
            exit_code=1,
            target="10.0.0.1",
            depth="standard",
            findings=[{"description": "test", "severity": "medium", "cve": None, "details": ""}],
            duration_seconds=60.0,
            cost_usd=0.10,
            tokens_used=5000,
            output_path="/tmp/results.sarif",
        )
        assert result.output_path == "/tmp/results.sarif"


class TestSARIFGenerator:
    """Verify SARIF v2.1.0 output structure and severity mapping."""

    def setup_method(self):
        self.generator = SARIFGenerator()

    def test_empty_findings(self):
        sarif = self.generator.generate([], "http://example.com")
        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert len(sarif["runs"]) == 1
        assert sarif["runs"][0]["results"] == []
        assert sarif["runs"][0]["tool"]["driver"]["name"] == "clearwing"

    def test_sarif_schema_structure(self):
        findings = [
            {
                "description": "Cross-Site Scripting",
                "severity": "high",
                "cve": "CVE-2023-9999",
                "details": "Reflected XSS in search parameter",
            }
        ]
        sarif = self.generator.generate(findings, "http://example.com")

        # Top-level keys
        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert "runs" in sarif

        # Run structure
        run = sarif["runs"][0]
        assert "tool" in run
        assert "results" in run

        # Tool structure
        driver = run["tool"]["driver"]
        assert "name" in driver
        assert "version" in driver
        assert "rules" in driver

        # Rule structure
        assert len(driver["rules"]) == 1
        rule = driver["rules"][0]
        assert rule["id"] == "CVE-2023-9999"
        assert "shortDescription" in rule
        assert rule["shortDescription"]["text"] == "Cross-Site Scripting"
        assert rule["fullDescription"]["text"] == "Reflected XSS in search parameter"

        # Result structure
        assert len(run["results"]) == 1
        result = run["results"][0]
        assert result["ruleId"] == "CVE-2023-9999"
        assert result["level"] == "error"
        assert result["message"]["text"] == "Cross-Site Scripting"
        assert len(result["locations"]) == 1
        assert (
            result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
            == "http://example.com"
        )

    def test_severity_mapping_critical(self):
        findings = [{"description": "RCE", "severity": "critical", "cve": None, "details": ""}]
        sarif = self.generator.generate(findings, "target")
        assert sarif["runs"][0]["results"][0]["level"] == "error"

    def test_severity_mapping_high(self):
        findings = [{"description": "SQLi", "severity": "high", "cve": None, "details": ""}]
        sarif = self.generator.generate(findings, "target")
        assert sarif["runs"][0]["results"][0]["level"] == "error"

    def test_severity_mapping_medium(self):
        findings = [{"description": "XSS", "severity": "medium", "cve": None, "details": ""}]
        sarif = self.generator.generate(findings, "target")
        assert sarif["runs"][0]["results"][0]["level"] == "warning"

    def test_severity_mapping_low(self):
        findings = [{"description": "Info leak", "severity": "low", "cve": None, "details": ""}]
        sarif = self.generator.generate(findings, "target")
        assert sarif["runs"][0]["results"][0]["level"] == "note"

    def test_severity_mapping_info(self):
        findings = [{"description": "Banner", "severity": "info", "cve": None, "details": ""}]
        sarif = self.generator.generate(findings, "target")
        assert sarif["runs"][0]["results"][0]["level"] == "note"

    def test_custom_tool_name(self):
        sarif = self.generator.generate([], "target", tool_name="my-scanner")
        assert sarif["runs"][0]["tool"]["driver"]["name"] == "my-scanner"

    def test_generated_rule_id_when_no_cve(self):
        findings = [
            {"description": "Issue A", "severity": "medium", "cve": None, "details": ""},
            {"description": "Issue B", "severity": "low", "cve": None, "details": ""},
        ]
        sarif = self.generator.generate(findings, "target")
        results = sarif["runs"][0]["results"]
        assert results[0]["ruleId"] == "VULN-0001"
        assert results[1]["ruleId"] == "VULN-0002"

    def test_multiple_findings_same_cve(self):
        findings = [
            {
                "description": "First instance",
                "severity": "high",
                "cve": "CVE-2024-0001",
                "details": "",
            },
            {
                "description": "Second instance",
                "severity": "high",
                "cve": "CVE-2024-0001",
                "details": "",
            },
        ]
        sarif = self.generator.generate(findings, "target")
        # Two results but only one rule (deduplicated)
        assert len(sarif["runs"][0]["results"]) == 2
        assert len(sarif["runs"][0]["tool"]["driver"]["rules"]) == 1

    def test_network_finding_falls_back_to_target(self):
        """A finding without 'file' uses target as artifactLocation.uri (regression)."""
        findings = [{"description": "Open port", "severity": "low", "cve": None, "details": ""}]
        sarif = self.generator.generate(findings, "192.168.1.1")
        loc = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "192.168.1.1"
        assert "region" not in loc

    def test_file_finding_uses_file_uri(self):
        """A finding with 'file' uses file path as artifactLocation.uri."""
        findings = [
            {
                "description": "Buffer overflow",
                "severity": "critical",
                "cve": None,
                "details": "",
                "file": "src/codec_a.c",
                "line_number": 47,
            }
        ]
        sarif = self.generator.generate(findings, "https://github.com/example/repo")
        loc = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "src/codec_a.c"
        assert loc["region"]["startLine"] == 47
        assert "endLine" not in loc["region"]

    def test_file_finding_with_end_line(self):
        """A finding with end_line emits a region with both startLine and endLine."""
        findings = [
            {
                "description": "SQL injection",
                "severity": "high",
                "cve": "CWE-89",
                "details": "",
                "file": "app/views.py",
                "line_number": 120,
                "end_line": 125,
            }
        ]
        sarif = self.generator.generate(findings, "repo")
        region = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
        assert region["startLine"] == 120
        assert region["endLine"] == 125

    def test_file_finding_without_line_number(self):
        """A finding with 'file' but no line_number uses file but no region."""
        findings = [
            {
                "description": "Hardcoded secret",
                "severity": "medium",
                "cve": None,
                "details": "",
                "file": "config/settings.py",
            }
        ]
        sarif = self.generator.generate(findings, "repo")
        loc = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "config/settings.py"
        assert "region" not in loc


class TestBuildGoal:
    """Verify _build_goal returns different messages per depth."""

    def _make_runner(self, depth: str) -> CICDRunner:
        return CICDRunner(target="192.168.1.1", depth=depth)

    def test_quick_goal(self):
        runner = self._make_runner("quick")
        goal = runner._build_goal()
        assert "192.168.1.1" in goal
        assert "quick" in goal.lower() or "fast" in goal.lower() or "top 100" in goal.lower()
        assert "CI/CD" in goal

    def test_standard_goal(self):
        runner = self._make_runner("standard")
        goal = runner._build_goal()
        assert "192.168.1.1" in goal
        assert "1000" in goal
        assert "CI/CD" in goal

    def test_deep_goal(self):
        runner = self._make_runner("deep")
        goal = runner._build_goal()
        assert "192.168.1.1" in goal
        assert "65535" in goal
        assert "CI/CD" in goal

    def test_goals_differ_by_depth(self):
        quick = self._make_runner("quick")._build_goal()
        standard = self._make_runner("standard")._build_goal()
        deep = self._make_runner("deep")._build_goal()
        # All three should be meaningfully different
        assert quick != standard
        assert standard != deep
        assert quick != deep

    def test_invalid_depth_raises(self):
        runner = self._make_runner("turbo")
        with pytest.raises(ValueError, match="Unknown depth"):
            runner._build_goal()


class TestDetermineExitCode:
    """Verify _determine_exit_code maps severities to correct exit codes."""

    def _runner(self) -> CICDRunner:
        return CICDRunner(target="192.168.1.1")

    def test_no_findings_returns_zero(self):
        assert self._runner()._determine_exit_code([]) == 0

    def test_info_only_returns_zero(self):
        findings = [{"severity": "info"}, {"severity": "info"}]
        assert self._runner()._determine_exit_code(findings) == 0

    def test_low_only_returns_zero(self):
        findings = [{"severity": "low"}]
        assert self._runner()._determine_exit_code(findings) == 0

    def test_medium_returns_one(self):
        findings = [{"severity": "medium"}]
        assert self._runner()._determine_exit_code(findings) == 1

    def test_medium_and_low_returns_one(self):
        findings = [{"severity": "medium"}, {"severity": "low"}]
        assert self._runner()._determine_exit_code(findings) == 1

    def test_high_returns_two(self):
        findings = [{"severity": "high"}]
        assert self._runner()._determine_exit_code(findings) == 2

    def test_critical_returns_two(self):
        findings = [{"severity": "critical"}]
        assert self._runner()._determine_exit_code(findings) == 2

    def test_mixed_critical_medium_returns_two(self):
        findings = [{"severity": "medium"}, {"severity": "critical"}, {"severity": "low"}]
        assert self._runner()._determine_exit_code(findings) == 2

    def test_missing_severity_treated_as_info(self):
        findings = [{"description": "something"}]
        assert self._runner()._determine_exit_code(findings) == 0

    def test_none_severity_treated_as_info(self):
        findings = [{"severity": None}]
        assert self._runner()._determine_exit_code(findings) == 0


class TestDriveWithAutoDecline:
    """Issue #20: an unattended run must auto-decline approval gates.

    Session 9526f073 aborted on its first gated tool (cve_db_update's
    download confirm) because nobody could answer the prompt. The drive
    loop now declines every pending gate and keeps the run going.
    """

    @pytest.mark.asyncio
    async def test_paused_gate_is_declined_and_run_completes(self):
        from types import SimpleNamespace

        from clearwing.agent.runtime import NativeAgentGraph
        from clearwing.agent.tooling import interrupt, tool
        from clearwing.llm.messages import AIMessage
        from clearwing.runners.cicd.runner import drive_with_auto_decline

        @tool(name="gated_tool")
        def gated_tool() -> dict:
            """Tool guarded by a human-approval gate."""
            if not interrupt("Approve?"):
                return {"gated": "denied"}
            return {"gated": "approved"}

        graph = NativeAgentGraph(
            llm=object(),
            native_tools=[],
            tools=[gated_tool],
            system_prompt_fn=lambda s: "sys",
            model_name="m",
            session_id=None,
            state_updater_fn=lambda *a, **k: {},
            knowledge_graph_populator_fn=None,
            input_guardrail_tool_names=frozenset(),
            output_guardrail_tool_names=frozenset(),
            enable_cost_tracker=False,
            enable_episodic_memory=False,
            enable_audit=False,
            enable_knowledge_graph=False,
            enable_input_guardrail=False,
            enable_output_guardrail=False,
            enable_event_bus=False,
            enable_context_summarizer=False,
        )

        async def fake_step(st):
            requested = any(
                getattr(m, "tool_calls", None) for m in st.get("messages", [])
            )
            if not requested:
                st.setdefault("messages", []).append(
                    AIMessage(
                        content="need gate",
                        tool_calls=[
                            SimpleNamespace(
                                fn_name="gated_tool", call_id="c1", fn_arguments={}
                            )
                        ],
                    )
                )
            else:
                st.setdefault("messages", []).append(AIMessage(content="done after gate"))
            return {}

        graph._aassistant_step = fake_step
        config = {"configurable": {"thread_id": "cicd-decline-1"}}

        await drive_with_auto_decline(
            graph, {"messages": []}, config, limits_exceeded=lambda: False
        )

        assert graph.get_state(config).next == ()
        messages = graph.get_state(config).values["messages"]
        tool_results = [
            m.content for m in messages if getattr(m, "type", "") == "tool"
        ]
        assert any("denied" in c for c in tool_results)
        assert any(getattr(m, "content", "") == "done after gate" for m in messages)

    @pytest.mark.asyncio
    async def test_limits_exceeded_stops_promptly(self):
        from clearwing.agent.runtime import NativeAgentGraph
        from clearwing.llm.messages import AIMessage
        from clearwing.runners.cicd.runner import drive_with_auto_decline

        graph = NativeAgentGraph(
            llm=object(),
            native_tools=[],
            tools=[],
            system_prompt_fn=lambda s: "sys",
            model_name="m",
            session_id=None,
            state_updater_fn=lambda *a, **k: {},
            knowledge_graph_populator_fn=None,
            input_guardrail_tool_names=frozenset(),
            output_guardrail_tool_names=frozenset(),
            enable_cost_tracker=False,
            enable_episodic_memory=False,
            enable_audit=False,
            enable_knowledge_graph=False,
            enable_input_guardrail=False,
            enable_output_guardrail=False,
            enable_event_bus=False,
            enable_context_summarizer=False,
        )
        config = {"configurable": {"thread_id": "cicd-decline-2"}}

        async def fake_step(st):
            st.setdefault("messages", []).append(AIMessage(content="one step"))
            return {}

        graph._aassistant_step = fake_step
        # Deadline already passed: the drive loop returns after the first
        # yielded event and never issues another assistant step.
        await drive_with_auto_decline(
            graph, {"messages": []}, config, limits_exceeded=lambda: True
        )
        messages = graph.get_state(config).values["messages"]
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_run_abandons_after_max_declines(self):
        """A model that keeps re-requesting gated tools must not keep the
        unattended run burning LLM calls until the wall-clock deadline."""
        from types import SimpleNamespace

        from clearwing.agent.runtime import NativeAgentGraph
        from clearwing.agent.tooling import interrupt, tool
        from clearwing.llm.messages import AIMessage
        from clearwing.runners.cicd.runner import drive_with_auto_decline

        @tool(name="gated_tool")
        def gated_tool() -> dict:
            """Tool guarded by a human-approval gate."""
            if not interrupt("Approve?"):
                return {"gated": "denied"}
            return {"gated": "approved"}

        graph = NativeAgentGraph(
            llm=object(),
            native_tools=[],
            tools=[gated_tool],
            system_prompt_fn=lambda s: "sys",
            model_name="m",
            session_id=None,
            state_updater_fn=lambda *a, **k: {},
            knowledge_graph_populator_fn=None,
            input_guardrail_tool_names=frozenset(),
            output_guardrail_tool_names=frozenset(),
            enable_cost_tracker=False,
            enable_episodic_memory=False,
            enable_audit=False,
            enable_knowledge_graph=False,
            enable_input_guardrail=False,
            enable_output_guardrail=False,
            enable_event_bus=False,
            enable_context_summarizer=False,
        )

        async def relentless_step(st):
            # Always re-requests the gate, decline after decline.
            st.setdefault("messages", []).append(
                AIMessage(
                    content="need gate",
                    tool_calls=[
                        SimpleNamespace(
                            fn_name="gated_tool", call_id=f"c{len(st['messages'])}", fn_arguments={}
                        )
                    ],
                )
            )
            return {}

        graph._aassistant_step = relentless_step
        config = {"configurable": {"thread_id": "cicd-decline-3"}}

        await drive_with_auto_decline(
            graph,
            {"messages": []},
            config,
            limits_exceeded=lambda: False,
            max_declines=3,
        )

        messages = graph.get_state(config).values["messages"]
        denied = [
            m
            for m in messages
            if getattr(m, "type", "") == "tool" and "denied" in m.content
        ]
        assert len(denied) == 3
