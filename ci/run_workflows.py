#!/usr/bin/env python3
"""Minimal ComfyUI workflow runner for CI smoke tests."""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

import requests
import websocket

DEFAULT_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
DEFAULT_CONNECT_TIMEOUT = int(os.environ.get("COMFYUI_CONNECT_TIMEOUT", "60"))
DEFAULT_WORKFLOW_TIMEOUT = int(os.environ.get("COMFYUI_WORKFLOW_TIMEOUT", "900"))


class ComfyWorkflowRunner:
    def __init__(self, host: str, port: int, connect_timeout: int, workflow_timeout: int, secure: bool = False) -> None:
        self.host = host
        self.port = port
        protocol_http = "https" if secure else "http"
        protocol_ws = "wss" if secure else "ws"
        self.base_http = f"{protocol_http}://{host}:{port}"
        self.base_ws = f"{protocol_ws}://{host}:{port}/ws"
        self.connect_timeout = connect_timeout
        self.workflow_timeout = workflow_timeout
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()
        self.websocket: Optional[websocket.WebSocket] = None

    def wait_for_server(self) -> None:
        deadline = time.monotonic() + self.connect_timeout
        while time.monotonic() < deadline:
            try:
                response = self.session.get(f"{self.base_http}/system_stats", timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(1)
        raise TimeoutError(f"ComfyUI server not reachable at {self.base_http}")

    def open_websocket(self) -> None:
        ws = websocket.WebSocket()
        ws.settimeout(5)
        ws.connect(f"{self.base_ws}?clientId={self.client_id}")
        self.websocket = ws

    def close_websocket(self) -> None:
        if self.websocket:
            try:
                self.websocket.close()
            finally:
                self.websocket = None

    def queue_prompt(self, prompt: dict) -> str:
        payload = {"prompt": prompt, "client_id": self.client_id}
        response = self.session.post(f"{self.base_http}/prompt", json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("No prompt_id returned from ComfyUI")
        return prompt_id

    def wait_for_completion(self, prompt_id: str) -> bool:
        if not self.websocket:
            raise RuntimeError("WebSocket connection not established")
        deadline = time.monotonic() + self.workflow_timeout
        ws = self.websocket
        while time.monotonic() < deadline:
            try:
                message = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"WebSocket error: {exc}", file=sys.stderr, flush=True)
                return False

            if isinstance(message, bytes):
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            message_type = payload.get("type")
            data = payload.get("data", {})

            if message_type == "execution_error":
                if data.get("prompt_id") == prompt_id:
                    print(f"Execution error: {payload}", file=sys.stderr, flush=True)
                    return False
            elif message_type == "status" and data.get("status") == "error":
                if data.get("prompt_id") == prompt_id:
                    print(f"Status error: {payload}", file=sys.stderr, flush=True)
                    return False
            elif message_type == "executing":
                if data.get("prompt_id") == prompt_id and data.get("node") is None:
                    return True
        print("Workflow timed out", file=sys.stderr, flush=True)
        return False

    def run_workflow(self, workflow_path: Path) -> bool:
        previous_workflow = os.environ.get("MGPU_JSON_WORKFLOW")
        previous_prompt = os.environ.get("MGPU_JSON_PROMPT")

        def restore_env() -> None:
            if previous_workflow is None:
                os.environ.pop("MGPU_JSON_WORKFLOW", None)
            else:
                os.environ["MGPU_JSON_WORKFLOW"] = previous_workflow
            if previous_prompt is None:
                os.environ.pop("MGPU_JSON_PROMPT", None)
            else:
                os.environ["MGPU_JSON_PROMPT"] = previous_prompt

        if workflow_path:
            os.environ["MGPU_JSON_WORKFLOW"] = workflow_path.name
        try:
            with workflow_path.open("r", encoding="utf-8") as handle:
                workflow = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to load workflow {workflow_path}: {exc}", file=sys.stderr, flush=True)
            restore_env()
            return False

        print(f"Running workflow {workflow_path}", flush=True)
        start = time.monotonic()
        try:
            prompt_id = self.queue_prompt(workflow)
            os.environ["MGPU_JSON_PROMPT"] = prompt_id
        except requests.HTTPError as exc:
            print(f"HTTP error while queueing workflow: {exc}", file=sys.stderr, flush=True)
            restore_env()
            return False
        except requests.RequestException as exc:
            print(f"Request error while queueing workflow: {exc}", file=sys.stderr, flush=True)
            restore_env()
            return False
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            restore_env()
            return False

        try:
            if not self.wait_for_completion(prompt_id):
                return False
            duration = time.monotonic() - start
            print(f"Workflow {workflow_path} completed in {duration:.2f}s", flush=True)
            return True
        finally:
            restore_env()

    def run_suite(self, workflows: Iterable[Path], fail_fast: bool) -> bool:
        self.wait_for_server()
        self.open_websocket()
        try:
            overall = True
            for workflow in workflows:
                ok = self.run_workflow(workflow)
                if not ok:
                    overall = False
                    if fail_fast:
                        break
            return overall
        finally:
            self.close_websocket()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ComfyUI workflows via the HTTP/WebSocket API")
    parser.add_argument("workflows", nargs="+", type=Path, help="Workflow files in ComfyUI API JSON format")
    parser.add_argument("--host", default=DEFAULT_HOST, help="ComfyUI HTTP host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ComfyUI HTTP port")
    parser.add_argument("--connect-timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT, help="Seconds to wait for the server to come online")
    parser.add_argument("--workflow-timeout", type=int, default=DEFAULT_WORKFLOW_TIMEOUT, help="Seconds to wait for each workflow to finish")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first workflow failure")
    parser.add_argument("--secure", action="store_true", help="Use secure HTTPS/WSS connections (default: insecure for localhost)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = ComfyWorkflowRunner(
        host=args.host,
        port=args.port,
        connect_timeout=args.connect_timeout,
        workflow_timeout=args.workflow_timeout,
        secure=args.secure,
    )
    success = runner.run_suite(args.workflows, fail_fast=args.fail_fast)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
