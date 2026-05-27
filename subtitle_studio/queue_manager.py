"""任务队列管理器：负责任务生命周期、并发调度、重试、暂停/恢复。"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from PySide6.QtCore import QObject, Signal

from .constants import MAX_RETRIES, PROGRESS_THROTTLE_MS, RETRY_BASE_DELAY, RETRYABLE_EXCEPTION_TYPES, RETRYABLE_STATUS_CODES
from .models import AppSettings, TaskCancelled, TaskState
from .orchestrator import TaskRunner
from .utils import new_task_id, normalize_path_key


# ──────────────────────────────────────────────────────────────
#  WorkerSignals — 跨线程 Qt 信号
# ──────────────────────────────────────────────────────────────

class WorkerSignals(QObject):
    progress = Signal(str, str, int, str)            # task_id, status, progress, message
    finished = Signal(str, bool, str, str, dict)      # task_id, success, status, message, outputs


# ──────────────────────────────────────────────────────────────
#  任务控制句柄（每个任务独享）
# ──────────────────────────────────────────────────────────────

class TaskControl:
    """单个任务的取消/暂停控制。"""

    __slots__ = ("cancel_event", "pause_event")

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # 初始为"运行态"（set = 不阻塞）


# ──────────────────────────────────────────────────────────────
#  可重试异常
# ──────────────────────────────────────────────────────────────

class RetryableError(Exception):
    """标记可重试的异常。"""


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得自动重试。"""
    if isinstance(exc, TaskCancelled):
        return False
    if isinstance(exc, RETRYABLE_EXCEPTION_TYPES):
        return True
    # HTTP 状态码判断
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status is not None:
        try:
            if int(status) in RETRYABLE_STATUS_CODES:
                return True
        except (TypeError, ValueError):
            pass
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    return False


# ──────────────────────────────────────────────────────────────
#  TaskQueueManager
# ──────────────────────────────────────────────────────────────

class TaskQueueManager(QObject):
    """集中管理任务队列：增删、启动/停止、暂停/恢复、单任务取消、重试、优先级排序。"""

    # 通知 UI 的信号
    task_progress = Signal(str, str, int, str)        # task_id, status, progress, message
    task_finished = Signal(str, bool, str, str, dict)  # task_id, success, status, message, outputs
    batch_finished = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.signals = WorkerSignals()

        # ── 任务存储 ──
        self.tasks: Dict[str, TaskState] = {}
        self.path_to_task: Dict[str, str] = {}

        # ── 运行时状态 ──
        self.active_run_ids: Set[str] = set()
        self.completed_run_ids: Set[str] = set()
        self.run_progress: Dict[str, int] = {}
        self.futures: Dict[str, Future] = {}
        self._controls: Dict[str, TaskControl] = {}

        # ── 线程池 ──
        self._executor: Optional[ThreadPoolExecutor] = None
        self.is_running = False

        # ── 进度节流 ──
        self._last_emit: Dict[str, float] = {}
        self._pending_progress: Dict[str, tuple[str, int, str]] = {}

        # ── 内部信号连接 ──
        self.signals.progress.connect(self._on_worker_progress)
        self.signals.finished.connect(self._on_worker_finished)

    # ──────────────────────────────────────────────────────────
    #  任务增删
    # ──────────────────────────────────────────────────────────

    def add_task(self, source_path: Path, row: int) -> Optional[str]:
        """添加任务，返回 task_id；重复则返回 None。"""
        key = normalize_path_key(source_path)
        if key in self.path_to_task:
            return None
        task_id = new_task_id()
        self.tasks[task_id] = TaskState(task_id=task_id, source_path=source_path, row=row)
        self.path_to_task[key] = task_id
        return task_id

    def remove_tasks(self, task_ids: List[str]) -> None:
        """移除指定任务（仅空闲时可调用）。"""
        for tid in task_ids:
            state = self.tasks.pop(tid, None)
            if state is not None:
                self.path_to_task.pop(normalize_path_key(state.source_path), None)

    def clear_all(self) -> None:
        """清空所有任务。"""
        self.tasks.clear()
        self.path_to_task.clear()
        self.active_run_ids.clear()
        self.completed_run_ids.clear()
        self.run_progress.clear()
        self.futures.clear()
        self._controls.clear()
        self._last_emit.clear()
        self._pending_progress.clear()

    def rebuild_row_mapping(self, path_to_row: Dict[str, int]) -> None:
        """表格行重建后同步 row 映射。"""
        for task in self.tasks.values():
            key = normalize_path_key(task.source_path)
            if key in path_to_row:
                task.row = path_to_row[key]

    # ──────────────────────────────────────────────────────────
    #  启动 / 停止批次
    # ──────────────────────────────────────────────────────────

    def start_batch(self, settings: AppSettings) -> int:
        """启动所有 Queued / Failed / Cancelled 任务，返回启动数量。"""
        if self.is_running:
            return 0
        run_ids = [
            tid for tid, t in self.tasks.items()
            if t.status in {"Queued", "Failed", "Cancelled"}
        ]
        if not run_ids:
            return 0

        self._executor = ThreadPoolExecutor(max_workers=settings.transcription.thread_count)
        self.active_run_ids = set(run_ids)
        self.completed_run_ids.clear()
        self.run_progress = {tid: 0 for tid in run_ids}
        self.futures.clear()
        self._controls.clear()
        self._last_emit.clear()
        self._pending_progress.clear()

        self.is_running = True

        # 按 priority 降序排列（priority 大的先执行）
        run_ids.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)

        for task_id in run_ids:
            task = self.tasks[task_id]
            task.status = "Queued"
            task.progress = 0
            task.message = "等待执行"
            task.start_time = 0.0
            task.end_time = 0.0
            task.error_detail = ""

            control = TaskControl()
            self._controls[task_id] = control

            future = self._executor.submit(
                self._worker_entry,
                task_id,
                task.source_path,
                settings,
                control,
            )
            self.futures[task_id] = future

        return len(run_ids)

    def stop_all(self) -> int:
        """请求停止所有任务，返回被取消的排队任务数。"""
        if not self.is_running:
            return 0
        canceled = 0
        for task_id, control in self._controls.items():
            control.cancel_event.set()
            control.pause_event.set()  # 唤醒暂停的线程以便取消
        for task_id, future in self.futures.items():
            if task_id in self.completed_run_ids:
                continue
            if future.cancel():
                canceled += 1
                self._mark_done(task_id, False, "Cancelled", "启动前已取消", {})
        return canceled

    # ──────────────────────────────────────────────────────────
    #  单任务操作
    # ──────────────────────────────────────────────────────────

    def cancel_task(self, task_id: str) -> None:
        """取消单个任务。"""
        control = self._controls.get(task_id)
        if control is None:
            return
        control.cancel_event.set()
        control.pause_event.set()  # 唤醒暂停线程

    def pause_task(self, task_id: str) -> None:
        """暂停单个运行中的任务。"""
        control = self._controls.get(task_id)
        if control is None or task_id not in self.active_run_ids or task_id in self.completed_run_ids:
            return
        control.pause_event.clear()
        state = self.tasks.get(task_id)
        if state and state.status not in {"Completed", "Failed", "Cancelled"}:
            state.status = "Paused"
            state.message = "已暂停"
            self.task_progress.emit(task_id, "Paused", state.progress, "已暂停")

    def resume_task(self, task_id: str) -> None:
        """恢复单个暂停的任务。"""
        control = self._controls.get(task_id)
        if control is None:
            return
        control.pause_event.set()
        state = self.tasks.get(task_id)
        if state and state.status == "Paused":
            state.status = "Transcribing"
            state.message = "恢复执行中"
            self.task_progress.emit(task_id, "Transcribing", state.progress, "恢复执行中")

    def retry_failed(self, task_ids: Optional[List[str]] = None) -> List[str]:
        """将失败/取消的任务重新标记为 Queued，返回重置的 task_id 列表。"""
        targets = task_ids or [
            tid for tid, t in self.tasks.items() if t.status in {"Failed", "Cancelled"}
        ]
        reset: List[str] = []
        for tid in targets:
            state = self.tasks.get(tid)
            if state and state.status in {"Failed", "Cancelled"}:
                state.status = "Queued"
                state.progress = 0
                state.message = "等待重试"
                state.error_detail = ""
                reset.append(tid)
        return reset

    # ──────────────────────────────────────────────────────────
    #  优先级操作
    # ──────────────────────────────────────────────────────────

    def set_priority(self, task_id: str, priority: int) -> None:
        state = self.tasks.get(task_id)
        if state:
            state.priority = priority

    def move_priority(self, task_ids: List[str], direction: int) -> None:
        """direction: +1=提升优先级, -1=降低优先级"""
        for tid in task_ids:
            state = self.tasks.get(tid)
            if state:
                state.priority += direction

    def reorder_by_rows(self, ordered_task_ids: List[str]) -> None:
        """按表格行顺序重新分配优先级（行号越大 priority 越高）。"""
        for idx, tid in enumerate(ordered_task_ids):
            state = self.tasks.get(tid)
            if state:
                state.priority = len(ordered_task_ids) - idx

    # ──────────────────────────────────────────────────────────
    #  查询
    # ──────────────────────────────────────────────────────────

    def get_selected_task_ids(self, rows: List[int]) -> List[str]:
        """根据表格行号获取 task_id 列表。"""
        row_to_tid: Dict[int, str] = {s.row: tid for tid, s in self.tasks.items()}
        return [row_to_tid[r] for r in rows if r in row_to_tid]

    def get_task_id_by_row(self, row: int) -> Optional[str]:
        for tid, s in self.tasks.items():
            if s.row == row:
                return tid
        return None

    def get_running_task_ids(self) -> List[str]:
        return [tid for tid in self.active_run_ids if tid not in self.completed_run_ids]

    def get_paused_task_ids(self) -> List[str]:
        return [tid for tid in self.active_run_ids if self.tasks.get(tid) and self.tasks[tid].status == "Paused"]

    def get_failed_task_ids(self) -> List[str]:
        return [tid for tid, t in self.tasks.items() if t.status == "Failed"]

    def get_summary(self) -> Dict[str, int]:
        total = len(self.tasks)
        queued = sum(1 for t in self.tasks.values() if t.status == "Queued")
        running = len(self.active_run_ids) - len(self.completed_run_ids)
        paused = sum(1 for t in self.tasks.values() if t.status == "Paused")
        done = sum(1 for t in self.tasks.values() if t.status == "Completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "Failed")
        canceled = sum(1 for t in self.tasks.values() if t.status == "Cancelled")
        return {"total": total, "queued": queued, "running": running, "paused": paused,
                "done": done, "failed": failed, "canceled": canceled}

    def get_total_progress(self) -> int:
        if not self.active_run_ids:
            return 0
        total = sum(self.run_progress.get(tid, 0) for tid in self.active_run_ids)
        return max(0, min(100, int(round(total / len(self.active_run_ids)))))

    # ──────────────────────────────────────────────────────────
    #  Worker 入口（带重试）
    # ──────────────────────────────────────────────────────────

    def _worker_entry(
        self,
        task_id: str,
        source_path: Path,
        settings: AppSettings,
        control: TaskControl,
    ) -> None:
        """在线程池中执行，包含重试逻辑。"""
        state = self.tasks.get(task_id)
        if state:
            state.start_time = time.monotonic()

        attempt = 0
        last_exc: Optional[BaseException] = None

        while attempt <= MAX_RETRIES:
            # 检查取消
            if control.cancel_event.is_set():
                self.signals.finished.emit(task_id, False, "Cancelled", "已取消", {})
                return

            # 重试前等待（指数退避）
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.signals.progress.emit(task_id, "Preparing", state.progress if state else 0,
                                           f"重试中 ({attempt}/{MAX_RETRIES})，等待 {delay:.0f}s...")
                # 分段等待以便及时响应取消
                waited = 0.0
                while waited < delay:
                    if control.cancel_event.is_set():
                        self.signals.finished.emit(task_id, False, "Cancelled", "等待重试时已取消", {})
                        return
                    sleep_chunk = min(0.5, delay - waited)
                    time.sleep(sleep_chunk)
                    waited += sleep_chunk

            try:
                runner = TaskRunner(settings)

                def report(status: str, progress: int, message: str) -> None:
                    # 暂停检查点
                    control.pause_event.wait()
                    if control.cancel_event.is_set():
                        raise TaskCancelled("已取消")
                    self._emit_progress_throttled(task_id, status, progress, message)

                outputs = runner.run_task(source_path, report, control.cancel_event)
                self.signals.finished.emit(task_id, True, "Completed", "完成", outputs)
                return

            except TaskCancelled as exc:
                self.signals.finished.emit(task_id, False, "Cancelled", str(exc), {})
                return

            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < MAX_RETRIES:
                    attempt += 1
                    continue
                # 不可重试或已达上限
                break

        # 所有重试用尽
        if last_exc is not None:
            message = str(last_exc).strip() or traceback.format_exc(limit=1)
            detail = traceback.format_exc(limit=5)
            self.signals.finished.emit(task_id, False, "Failed", message, {"_detail": detail})
        else:
            self.signals.finished.emit(task_id, False, "Failed", "未知错误", {})

    # ──────────────────────────────────────────────────────────
    #  进度节流
    # ──────────────────────────────────────────────────────────

    def _emit_progress_throttled(self, task_id: str, status: str, progress: int, message: str) -> None:
        now = time.monotonic()
        last = self._last_emit.get(task_id, 0.0)
        elapsed_ms = (now - last) * 1000

        self._pending_progress[task_id] = (status, progress, message)

        if elapsed_ms >= PROGRESS_THROTTLE_MS:
            self._last_emit[task_id] = now
            self.signals.progress.emit(task_id, status, progress, message)
        # 否则暂存，由 _on_worker_progress 中的定时器在下次刷新时取出

    def flush_pending_progress(self) -> None:
        """由 QTimer 定期调用，刷新暂存的进度信号。"""
        now = time.monotonic()
        for task_id, (status, progress, message) in list(self._pending_progress.items()):
            last = self._last_emit.get(task_id, 0.0)
            if (now - last) * 1000 >= PROGRESS_THROTTLE_MS:
                self._last_emit[task_id] = now
                self.signals.progress.emit(task_id, status, progress, message)
                self._pending_progress.pop(task_id, None)

    # ──────────────────────────────────────────────────────────
    #  信号处理（UI 线程）
    # ──────────────────────────────────────────────────────────

    def _on_worker_progress(self, task_id: str, status: str, progress: int, message: str) -> None:
        if task_id not in self.active_run_ids:
            return
        self.run_progress[task_id] = max(0, min(100, progress))
        state = self.tasks.get(task_id)
        if state:
            state.status = status
            state.progress = progress
            state.message = message
        self.task_progress.emit(task_id, status, progress, message)

    def _on_worker_finished(self, task_id: str, success: bool, status: str, message: str, outputs: dict) -> None:
        self._mark_done(task_id, success, status, message, outputs)

    def _mark_done(self, task_id: str, success: bool, status: str, message: str, outputs: dict) -> None:
        if task_id not in self.active_run_ids or task_id in self.completed_run_ids:
            return
        self.completed_run_ids.add(task_id)
        state = self.tasks.get(task_id)
        if state:
            state.end_time = time.monotonic()
            state.progress = 100 if success else state.progress
            state.status = status
            state.message = message
            state.outputs = outputs
            if not success and "_detail" in outputs:
                state.error_detail = outputs.pop("_detail", "")

        self.run_progress[task_id] = 100 if success else self.run_progress.get(task_id, 0)

        self.task_finished.emit(task_id, success, status, message, outputs)

        if len(self.completed_run_ids) >= len(self.active_run_ids):
            self._finish_batch()

    def _finish_batch(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._executor = None
        self.is_running = False
        self._pending_progress.clear()
        self.batch_finished.emit()
