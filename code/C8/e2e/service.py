from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


class LiveServiceProcess:
    def __init__(
        self,
        *,
        project_dir: Path,
        host: str,
        port: int,
        model: str,
        reuse_server: bool,
    ):
        self.project_dir = project_dir
        self.host = host
        self.port = port
        self.model = model
        self.reuse_server = reuse_server
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if is_port_open(self.host, self.port):
            if self.reuse_server:
                return
            raise RuntimeError(f"port {self.host}:{self.port} is already in use; pass --reuse-server to use it")

        env = os.environ.copy()
        env["RAG_LLM_MODEL"] = self.model
        env["FLASK_RUN_HOST"] = self.host
        env["FLASK_RUN_PORT"] = str(self.port)
        command = [
            sys.executable,
            "-c",
            (
                "from web_app import create_app; "
                "app=create_app(); "
                f"app.run(host='{self.host}', port={self.port}, debug=False, threaded=True)"
            ),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(self.project_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
