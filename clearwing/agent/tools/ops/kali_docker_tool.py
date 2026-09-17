import platform
import re

from clearwing.agent.tooling import current_session_id, interrupt, tool

CONTAINER_NAME = "clearwing-kali"
# Host-side artifacts are bind-mounted here (issue #13): anything the tools
# write to this path survives container removal. Without the mount, scan and
# exploit artifacts died with the container and were unrecoverable from the
# host.
ARTIFACTS_MOUNT = "/artifacts"

# Docker container names and host paths both take this charset; anything
# else (a tampered --resume id, a foreign caller) falls back to the shared
# adhoc scope instead of traversing or injecting.
_SCOPE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _kali_scope(session_id: str | None) -> str:
    """Return the container/artifacts scope label for a session.

    Uses the FULL session id — truncating to 8 chars would cut hunt ids
    (``sh-<hex8>``) down to ~20 bits of entropy and let concurrent hunts
    collide on both the container name and the artifacts dir. Containers
    are named per session so one session's installed tools and artifacts
    never bleed into another's; session-less callers (CLI scripts) share
    the legacy ``adhoc`` scope.
    """
    if session_id and _SCOPE_PATTERN.match(session_id):
        return session_id
    return "adhoc"


def _artifacts_dir(session_id: str | None):
    from clearwing.core.config import clearwing_home

    scope = _kali_scope(session_id)
    return clearwing_home() / "kali" / scope / "artifacts"


def _has_artifacts_mount(container, expected_source: str | None = None) -> bool:
    """True when the container carries a usable /artifacts bind mount.

    A mount is only usable when it is a WRITABLE BIND from the artifact dir
    this session computed — a named volume, a different host source (e.g.
    CLEARWING_HOME changed between runs), or a read-only mount stores
    outputs elsewhere or fails them outright, so promising persistence
    would be wrong (Codex PR-55 r2).
    """
    try:
        attrs = container.attrs or {}
        for mount in attrs.get("Mounts", []):
            if mount.get("Destination") != ARTIFACTS_MOUNT:
                continue
            if mount.get("Type") != "bind":
                return False
            if mount.get("RW") is False:
                return False
            source = mount.get("Source")
            if expected_source and source and str(source) != expected_source:
                return False
            return True
    except Exception:
        return False
    return False


@tool
def kali_setup() -> dict:
    """Start a Kali Linux Docker container for specialized security tools.

    Pulls kalilinux/kali-rolling if not present, starts a container, and
    returns the container ID. Reuses existing container if one is already
    running. A host directory is bind-mounted at ``/artifacts`` (issue #13):
    write every durable output (scan results, exploit artifacts, notes) to
    that path — files there persist on the host after the container is
    removed, unlike the container filesystem.

    Returns:
        Dict with keys: container_id, status, message, artifacts_dir
        (host-side path), artifacts_mount (in-container path).
    """
    import docker

    session_id = current_session_id()
    scope = _kali_scope(session_id)
    name = f"clearwing-kali-{scope}" if scope != "adhoc" else CONTAINER_NAME
    artifacts_dir = _artifacts_dir(session_id)

    client = docker.from_env()

    # Check for existing container
    try:
        existing = client.containers.get(name)
        mounted = _has_artifacts_mount(existing, str(artifacts_dir))
        base = {
            "container_id": existing.id,
            "artifacts_dir": str(artifacts_dir),
        }
        if existing.status == "running":
            if not mounted:
                # A pre-upgrade container cannot gain a bind mount after
                # creation — promising /artifacts persistence here would
                # repeat the data loss #13 was filed for.
                return {
                    **base,
                    "status": "reused",
                    "artifacts_mount": None,
                    "message": (
                        f"Reusing existing Kali container {existing.short_id} "
                        "(LEGACY: no /artifacts mount — files written inside "
                        "will NOT persist; run kali_cleanup then kali_setup "
                        "to get a mounted container)"
                    ),
                }
            return {
                **base,
                "status": "reused",
                "artifacts_mount": ARTIFACTS_MOUNT,
                "message": f"Reusing existing Kali container {existing.short_id}",
            }
        existing.start()
        if not mounted:
            # A stopped legacy container restarts equally mount-less — same
            # honest degraded warning as the running-legacy branch (Codex
            # PR-55 P2: silently restarting it kept losing artifacts).
            return {
                **base,
                "status": "restarted",
                "artifacts_mount": None,
                "message": (
                    f"Restarted existing Kali container {existing.short_id} "
                    "(LEGACY: no /artifacts mount — files written inside "
                    "will NOT persist; run kali_cleanup then kali_setup "
                    "to get a mounted container)"
                ),
            }
        return {
            **base,
            "status": "restarted",
            "artifacts_mount": ARTIFACTS_MOUNT,
            "message": f"Restarted existing Kali container {existing.short_id}",
        }
    except docker.errors.NotFound:
        pass

    # Pull image if needed
    try:
        client.images.get("kalilinux/kali-rolling")
    except docker.errors.ImageNotFound:
        client.images.pull("kalilinux/kali-rolling")

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    network_mode = "host" if platform.system() == "Linux" else "bridge"

    container = client.containers.run(
        "kalilinux/kali-rolling",
        command="sleep infinity",
        name=name,
        network_mode=network_mode,
        volumes={str(artifacts_dir): {"bind": ARTIFACTS_MOUNT, "mode": "rw"}},
        detach=True,
        tty=True,
    )

    return {
        "container_id": container.id,
        "status": "created",
        "artifacts_dir": str(artifacts_dir),
        "artifacts_mount": ARTIFACTS_MOUNT,
        "message": f"Started new Kali container {container.short_id}",
    }


@tool
def kali_execute(container_id: str, command: str) -> dict:
    """Execute a command inside the Kali Docker container. REQUIRES HUMAN APPROVAL.

    Durable outputs (scan results, exploit artifacts) must be written under
    /artifacts to survive container removal — see kali_setup.

    Args:
        container_id: Docker container ID.
        command: Shell command to execute.

    Returns:
        Dict with keys: exit_code, output.
    """
    approval = interrupt(f"Approve running in Kali container: {command}")
    if not approval:
        return {"exit_code": -1, "output": "Command denied by user"}

    import docker

    client = docker.from_env()
    container = client.containers.get(container_id)
    exit_code, output = container.exec_run(command, tty=True)
    return {
        "exit_code": exit_code,
        "output": output.decode("utf-8", errors="replace"),
    }


@tool
def kali_install_tool(container_id: str, package_name: str) -> dict:
    """Install a package in the Kali Docker container via apt-get.

    REQUIRES HUMAN APPROVAL: the package name reaches a shell command
    inside the container, and installed tooling is dual-use.

    Args:
        container_id: Docker container ID.
        package_name: Debian package name(s) to install (e.g. 'nmap',
            'nikto'). Only letters, digits, '.', '+', '-', '_' and spaces
            are accepted — shell metacharacters are rejected.

    Returns:
        Dict with keys: exit_code, output. exit_code -2 means the package
        name was rejected or the command denied.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9.+_ -]*", package_name or ""):
        return {
            "exit_code": -2,
            "output": (
                f"Rejected package_name {package_name!r}: only Debian package "
                "names (letters, digits, '.', '+', '_', '-', spaces) are allowed"
            ),
        }
    approval = interrupt(f"Approve installing Kali package(s): {package_name}")
    if not approval:
        return {"exit_code": -2, "output": "Package install denied by user"}

    import docker

    client = docker.from_env()
    container = client.containers.get(container_id)
    exit_code, output = container.exec_run(
        f"apt-get update -qq && apt-get install -y -qq {package_name}", tty=True
    )
    return {
        "exit_code": exit_code,
        "output": output.decode("utf-8", errors="replace"),
    }


@tool
def kali_cleanup(container_id: str) -> dict:
    """Stop and remove the Kali Docker container.

    Files written under /artifacts persist on the host after removal — the
    artifacts dir is returned by kali_setup and is NOT deleted here.

    Args:
        container_id: Docker container ID.

    Returns:
        Dict with keys: status, message.
    """
    import docker

    client = docker.from_env()
    try:
        container = client.containers.get(container_id)
        container.stop(timeout=5)
        container.remove()
        return {"status": "removed", "message": f"Container {container_id[:12]} removed"}
    except docker.errors.NotFound:
        return {"status": "not_found", "message": "Container not found"}
