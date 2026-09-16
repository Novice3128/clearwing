"""Tests for the agent graph updates — flag detection, state expansion, guardrail integration."""

import json

from clearwing.agent.prompts import build_dynamic_context, build_system_prompt
from clearwing.agent.runtime import FLAG_PATTERNS, detect_flags
from clearwing.agent.state import AgentState
from clearwing.agent.tools import get_all_tools


class TestAgentState:
    def test_state_has_all_fields(self):
        """Verify AgentState TypedDict has all required keys."""
        annotations = AgentState.__annotations__
        expected_fields = {
            "messages",
            "target",
            "open_ports",
            "services",
            "vulnerabilities",
            "exploit_results",
            "os_info",
            "kali_container_id",
            "custom_tool_names",
            # Phase 1 additions:
            "session_id",
            "flags_found",
            "loaded_skills",
            "paused",
            "total_cost_usd",
            "total_tokens",
        }
        assert expected_fields.issubset(set(annotations.keys())), (
            f"Missing fields: {expected_fields - set(annotations.keys())}"
        )


class TestFlagDetection:
    def test_detect_flag_curly_braces(self):
        flags = detect_flags("Found flag{this_is_a_test_flag}")
        assert len(flags) >= 1
        assert any("flag{this_is_a_test_flag}" in f["flag"] for f in flags)

    def test_detect_flag_uppercase(self):
        flags = detect_flags("Got FLAG{UPPERCASE_FLAG}")
        assert len(flags) >= 1

    def test_detect_htb_flag(self):
        flags = detect_flags("The flag is HTB{hackthebox_flag_123}")
        assert len(flags) >= 1
        assert any("HTB{hackthebox_flag_123}" in f["flag"] for f in flags)

    def test_detect_ctf_flag(self):
        flags = detect_flags("CTF{capture_the_flag}")
        assert len(flags) >= 1

    def test_detect_md5_hash(self):
        flags = detect_flags("Hash: d41d8cd98f00b204e9800998ecf8427e")
        assert len(flags) >= 1

    def test_no_flags_in_clean_text(self):
        flags = detect_flags("Port 22 is open running OpenSSH")
        assert len(flags) == 0

    def test_multiple_flags(self):
        text = "Found flag{first} and also FLAG{second}"
        flags = detect_flags(text)
        flag_values = {f["flag"] for f in flags}
        assert "flag{first}" in flag_values
        assert "FLAG{second}" in flag_values

    def test_flag_patterns_count(self):
        assert len(FLAG_PATTERNS) >= 4

    def test_64_hex_container_id_is_not_two_flags(self):
        # Issue #35: a 64-hex container id used to match the bare 32-hex
        # pattern twice (finditer windows) and inflate flags_found.
        container_id = "a" * 64
        flags = detect_flags(f'{{"container_id": "{container_id}"}}')
        assert flags == []

    def test_standalone_32_hex_still_matches(self):
        flags = detect_flags("Hash: d41d8cd98f00b204e9800998ecf8427e")
        assert [f["flag"] for f in flags] == ["d41d8cd98f00b204e9800998ecf8427e"]

    def test_cross_pattern_duplicates_are_deduped(self):
        # "FLAG{...}" matches both the case-insensitive flag{} pattern and
        # the exact-case FLAG{} pattern — one capture, one entry.
        flags = detect_flags("Got FLAG{UPPERCASE_FLAG}")
        assert [f["flag"] for f in flags] == ["FLAG{UPPERCASE_FLAG}"]

    def test_structured_tool_output_with_container_id_yields_no_flag(self):
        # Issue #35: kali tools return {"container_id": <64-hex>}; the
        # tool-scan path masks known id fields before scanning, so the id
        # never pollutes flags_found (while real loot still matches).
        from clearwing.agent.runtime import _strip_id_fields

        payload = {
            "container_id": "f" * 64,
            "kali_container_id": "e" * 64,
            "nested": [{"image_id": "d" * 64, "note": "clean"}],
            "output": "flag{real_loot_1}",
        }
        scan_text = json.dumps(_strip_id_fields(payload))
        flags = detect_flags(scan_text)
        assert [f["flag"] for f in flags] == ["flag{real_loot_1}"]


class TestGetAllTools:
    def test_tools_count(self):
        tools = get_all_tools()
        # 22 original + 4 new memory/skills tools = 26
        assert len(tools) >= 26

    def test_new_tools_present(self):
        tools = get_all_tools()
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "recall_target_history" in tool_names
        assert "store_knowledge" in tool_names
        assert "search_knowledge" in tool_names
        assert "load_skills" in tool_names


class TestBuildSystemPrompt:
    def test_prompt_includes_skills_section(self):
        state = {
            "target": "10.0.0.1",
            "open_ports": [],
            "services": [],
            "vulnerabilities": [],
            "exploit_results": [],
            "os_info": None,
            "kali_container_id": None,
            "custom_tool_names": [],
            "flags_found": [],
            "loaded_skills": [],
        }
        prompt = build_system_prompt(state)
        assert "Skills System" in prompt or "skills" in prompt.lower()

    def test_prompt_includes_flags(self):
        state = {
            "target": "10.0.0.1",
            "open_ports": [],
            "services": [],
            "vulnerabilities": [],
            "exploit_results": [],
            "os_info": None,
            "kali_container_id": None,
            "custom_tool_names": [],
            "flags_found": [{"flag": "flag{test}", "pattern": ".*"}],
            "loaded_skills": [],
        }
        # State-derived content rides the dynamic context note (#36), not
        # the static system prompt.
        note = build_dynamic_context(state)
        assert "flag{test}" in note
        assert "flag{test}" not in build_system_prompt(state)

    def test_prompt_includes_target(self):
        state = {
            "target": "192.168.1.100",
            "open_ports": [],
            "services": [],
            "vulnerabilities": [],
            "exploit_results": [],
            "os_info": None,
            "kali_container_id": None,
            "custom_tool_names": [],
            "flags_found": [],
            "loaded_skills": [],
        }
        note = build_dynamic_context(state)
        assert "192.168.1.100" in note
        assert "192.168.1.100" not in build_system_prompt(state)
