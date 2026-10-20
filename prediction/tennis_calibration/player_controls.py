"""Native companion controls for the MuJoCo tennis replay.

Tk widgets never read or mutate MuJoCo objects. Their only effect is enqueueing
small command dictionaries for the replay's main loop. Create and poll this
window on the main thread; ``poll`` takes an ordinary UI-state snapshot.
Trajectory identities always use the full ZIP member path, including when
different acquisition folders contain the same filename.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence


SPEED_PRESETS = (.1, .25, .5, 1., 2., 4.)
SPLIT_LABELS = {"all": "全部", "train": "训练", "validation": "验证", "test": "测试"}
LAYER_LABELS = {
    "reference": "灰色 · 测量参考",
    "received": "绿色 · 已收到观测",
    "estimate": "蓝色 · 在线估计",
    "forecast": "橙色 · 未来预测",
    "ball": "黄色 · 自由仿真球",
    "court": "球场背景",
}
CAMERA_LABELS = {"overview": "全景", "top": "俯视", "side": "侧视", "follow": "跟随球"}
VIEW_LABELS = {"online": "在线滚动预测", "bounce": "触地前冻结 · 反弹预测"}
BOUNCE_MODEL_LABELS = {
    "fixed": "固定反弹（物理）",
    "ridge1": "线性修正（Ridge）",
    "ridge2": "二次修正（Ridge）",
    "gp": "GP 修正（本次最优）",
}
BOUNCE_GAPS = (0., .005, .01, .02, .04, .08)
BOUNCE_LAYER_LABELS = {
    **LAYER_LABELS,
    "received": "绿色 · 冻结前观测",
    "estimate": "蓝色 · 所选模型预测",
    "forecast": "橙色 · 冻结输入预测",
    "ball": "黄色 · 所选反弹仿真",
}


def _key_from_label(labels, value, default):
    return next((key for key, label in labels.items() if label == value), default)


def clamp_speed(value: float) -> float:
    """The native replay has an explicit 0.1x minimum and 4x maximum."""
    return min(4., max(.1, float(value)))


def trajectory_choices(traces: Sequence[Mapping], split="all", search=""):
    """Return stable global IDs, full identities, and readable menu labels."""
    query = str(search).strip().casefold()
    choices = []
    for index, trace in enumerate(traces):
        identity = str(trace["id"])
        group = str(trace.get("split", ""))
        if split != "all" and split != group:
            continue
        label = f"{index + 1:03d} · {SPLIT_LABELS.get(group, group)} · {identity}"
        if query and query not in label.casefold() and query not in group.casefold():
            continue
        choices.append({"index": index, "trajectory_id": identity, "split": group, "label": label})
    return choices


class ControlsWindow:
    """Companion native window; callbacks enqueue commands and return quickly.

    ``state`` keys: trajectory_id, paused, speed, loop, auto_next, elapsed_s,
    duration_s, raw_rows_received, forecast_updates, visible (layer->bool),
    camera, view_mode, bounce_model, bounce_gap_s, bounce_compare,
    bounce_status. ``poll(state)`` refreshes labels/widgets and processes Tk events.
    Closing the panel emits ``panel_close``; the replay loop handles shutdown.
    """

    def __init__(self, traces, enqueue: Callable[[dict], None], state=None,
                 *, title="网球轨迹 · MuJoCo 播放控制", geometry=None):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.traces = list(traces)
        self.enqueue = enqueue
        self.closed = False
        self._suspend = True
        self._speed_dragging = self._seek_dragging = False
        self._state = dict(state or {})
        self._choices = []
        self._last_snapshot = None
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(background="#111d2b")
        self.root.minsize(510, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        # Explicit main-thread pumping is compatible with the passive native
        # MuJoCo viewer; no Tk thread or second mainloop is introduced.
        width, height = 540, min(940, self.root.winfo_screenheight() - 90)
        self.root.geometry(geometry or f"{width}x{height}+{max(0, self.root.winfo_screenwidth()-width-24)}+44")
        families = set(__import__("tkinter.font", fromlist=["families"]).families(self.root))
        font_family = next((name for name in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans")
                            if name in families), "TkDefaultFont")
        self.font_family = font_family
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#111d2b")
        style.configure("TLabel", background="#111d2b", foreground="#e2ebf5", font=(font_family, 10))
        style.configure("Title.TLabel", font=(font_family, 16, "bold"), foreground="#ffffff")
        style.configure("Muted.TLabel", foreground="#9ab0c7", font=(font_family, 9))
        style.configure("TLabelframe", background="#111d2b", bordercolor="#34495e")
        style.configure("TLabelframe.Label", background="#111d2b", foreground="#92d9bd", font=(font_family, 10, "bold"))
        style.configure("TButton", font=(font_family, 10), padding=(9, 6), background="#263e52", foreground="#f0f5fa")
        style.map("TButton", background=[("active", "#386079")])
        style.configure("TCheckbutton", background="#111d2b", foreground="#e2ebf5", font=(font_family, 10))
        style.map("TCheckbutton", background=[("active", "#1e3042")])
        style.configure("TCombobox", font=(font_family, 10), padding=4)
        style.configure("Horizontal.TScale", background="#111d2b", troughcolor="#304257")
        self.root.option_add("*TCombobox*Listbox.font", (font_family, 10))

        shell = ttk.Frame(self.root, padding=(12, 12))
        shell.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(shell, highlightthickness=0, borderwidth=0,
                                background="#111d2b")
        self.scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y", padx=(5, 0))
        self.canvas.pack(side="left", fill="both", expand=True)
        outer = ttk.Frame(self.canvas, padding=(5, 4, 5, 12))
        self.content = outer
        self._content_window = self.canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self._content_window, width=event.width))
        self.root.bind("<MouseWheel>", self._scroll_wheel)
        self.root.bind("<Button-4>", self._scroll_wheel)
        self.root.bind("<Button-5>", self._scroll_wheel)
        ttk.Label(outer, text="网球轨迹回放", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="真实观测 · 在线预测 · 冻结输入反弹对照", style="Muted.TLabel").pack(anchor="w", pady=(2, 12))

        prediction = ttk.LabelFrame(outer, text="观察内容", padding=12)
        prediction.pack(fill="x", pady=(0, 10))
        self.view_var = tk.StringVar(value=VIEW_LABELS["online"])
        self.view_box = ttk.Combobox(prediction, textvariable=self.view_var,
            values=list(VIEW_LABELS.values()), state="readonly")
        self.view_box.pack(fill="x")
        self.view_box.bind("<<ComboboxSelected>>", lambda event: self._emit("view_mode",
            value=_key_from_label(VIEW_LABELS, self.view_var.get(), "online")))
        self.view_help_var = tk.StringVar()
        ttk.Label(prediction, textvariable=self.view_help_var, style="Muted.TLabel",
            wraplength=455).pack(anchor="w", pady=(5, 7))
        bounce_row = ttk.Frame(prediction)
        bounce_row.pack(fill="x")
        ttk.Label(bounce_row, text="反弹模型").pack(side="left", padx=(0, 7))
        self.bounce_model_var = tk.StringVar(value=BOUNCE_MODEL_LABELS["gp"])
        self.bounce_model_box = ttk.Combobox(bounce_row, textvariable=self.bounce_model_var,
            values=list(BOUNCE_MODEL_LABELS.values()), state="disabled")
        self.bounce_model_box.pack(side="left", fill="x", expand=True)
        self.bounce_model_box.bind("<<ComboboxSelected>>", lambda event: self._emit("bounce_model",
            value=_key_from_label(BOUNCE_MODEL_LABELS, self.bounce_model_var.get(), "gp")))
        gap_row = ttk.Frame(prediction)
        gap_row.pack(fill="x", pady=(7, 0))
        ttk.Label(gap_row, text="缺失触地前末段").pack(side="left", padx=(0, 7))
        self.bounce_gap_var = tk.StringVar(value="0 ms")
        self.bounce_gap_box = ttk.Combobox(gap_row, textvariable=self.bounce_gap_var,
            values=[f"{round(gap*1000)} ms" for gap in BOUNCE_GAPS],
            state="disabled", width=7)
        self.bounce_gap_box.pack(side="left")
        self.bounce_gap_box.bind("<<ComboboxSelected>>", self._bounce_gap_selected)
        self.bounce_compare_var = tk.BooleanVar(value=True)
        self.bounce_compare_button = ttk.Checkbutton(prediction,
            text="叠加固定反弹模型对照", variable=self.bounce_compare_var,
            command=lambda: self._emit("bounce_compare", enabled=self.bounce_compare_var.get()))
        self.bounce_compare_button.pack(anchor="w", pady=(7, 0))
        self.bounce_status_var = tk.StringVar()
        ttk.Label(prediction, textvariable=self.bounce_status_var,
            style="Muted.TLabel", wraplength=455).pack(anchor="w", pady=(4, 0))

        sample = ttk.LabelFrame(outer, text="样本轨迹", padding=12)
        sample.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(sample)
        row.pack(fill="x")
        self.split_var = tk.StringVar(value=SPLIT_LABELS.get(self._state.get("selection_split", "all"), "全部"))
        self.split_box = ttk.Combobox(row, textvariable=self.split_var,
            values=list(SPLIT_LABELS.values()), state="readonly", width=7)
        self.split_box.pack(side="left", padx=(0, 8))
        self.split_box.bind("<<ComboboxSelected>>", lambda event: self._filter_choices())
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(row, textvariable=self.search_var)
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", lambda event: self._filter_choices())
        ttk.Button(row, text="清空", command=self._clear_search).pack(side="left", padx=(6, 0))
        ttk.Label(sample, text="搜索序号、文件夹或文件名", style="Muted.TLabel").pack(anchor="w", pady=(5, 4))
        self.trace_var = tk.StringVar()
        self.trace_box = ttk.Combobox(sample, textvariable=self.trace_var, state="readonly")
        self.trace_box.pack(fill="x")
        self.trace_box.bind("<<ComboboxSelected>>", lambda event: self._choose())
        self.identity_var = tk.StringVar()
        ttk.Label(sample, textvariable=self.identity_var, wraplength=470, style="Muted.TLabel").pack(anchor="w", pady=(7, 4))
        nav = ttk.Frame(sample)
        nav.pack(fill="x")
        self.previous_button = ttk.Button(nav, text="← 上一条", command=lambda: self._move(-1))
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(nav, text="下一条 →", command=lambda: self._move(1))
        self.next_button.pack(side="left", padx=7)
        self.count_var = tk.StringVar()
        ttk.Label(nav, textvariable=self.count_var, style="Muted.TLabel").pack(side="right")

        playback = ttk.LabelFrame(outer, text="播放", padding=12)
        playback.pack(fill="x", pady=(0, 10))
        buttons = ttk.Frame(playback)
        buttons.pack(fill="x")
        self.pause_button = ttk.Button(buttons, text="暂停", command=self._pause)
        self.pause_button.pack(side="left")
        self.restart_button = ttk.Button(buttons, text="从头播放", command=lambda: self._emit("restart"))
        self.restart_button.pack(side="left", padx=7)
        self.loop_var, self.auto_next_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        ttk.Checkbutton(buttons, text="循环", variable=self.loop_var,
            command=lambda: self._emit("loop", enabled=self.loop_var.get())).pack(side="left", padx=(4, 7))
        ttk.Checkbutton(buttons, text="自动下一条", variable=self.auto_next_var,
            command=lambda: self._emit("auto_next", enabled=self.auto_next_var.get())).pack(side="left")
        self.progress_var = tk.DoubleVar(value=0.)
        self.progress_scale = ttk.Scale(playback, from_=0., to=1., variable=self.progress_var,
                                        command=self._progress_preview)
        self.progress_scale.pack(fill="x", pady=(12, 1))
        self.progress_scale.bind("<ButtonPress-1>", self._seek_press)
        self.progress_scale.bind("<ButtonRelease-1>", self._seek_release)
        self.progress_scale.bind("<KeyRelease>", self._seek_release)
        self.time_var = tk.StringVar(value="0.000 / 0.000 秒")
        ttk.Label(playback, textvariable=self.time_var, style="Muted.TLabel").pack(anchor="e")
        speedrow = ttk.Frame(playback)
        speedrow.pack(fill="x", pady=(8, 4))
        ttk.Label(speedrow, text="播放速度").pack(side="left")
        self.speed_label = tk.StringVar(value="0.50×")
        ttk.Label(speedrow, textvariable=self.speed_label).pack(side="right")
        self.speed_var = tk.DoubleVar(value=.5)
        self.speed_scale = ttk.Scale(playback, from_=.1, to=4., variable=self.speed_var,
                                     command=self._speed_changed)
        self.speed_scale.pack(fill="x")
        self.speed_scale.bind("<ButtonPress-1>", lambda event: setattr(self, "_speed_dragging", True))
        self.speed_scale.bind("<ButtonRelease-1>", self._speed_release)
        presets = ttk.Frame(playback)
        presets.pack(fill="x", pady=(6, 0))
        self.speed_buttons = {}
        for speed in SPEED_PRESETS:
            button = ttk.Button(presets, text=f"{speed:g}×", width=5,
                command=lambda value=speed: self._set_speed(value))
            button.pack(side="left", expand=True, fill="x", padx=2)
            self.speed_buttons[speed] = button

        display = ttk.LabelFrame(outer, text="显示与视角", padding=12)
        display.pack(fill="x", pady=(0, 10))
        self.layer_vars, self.layer_buttons = {}, {}
        for index, (name, label) in enumerate(LAYER_LABELS.items()):
            variable = tk.BooleanVar(value=True)
            self.layer_vars[name] = variable
            button = ttk.Checkbutton(display, text=label, variable=variable,
                command=lambda layer=name, var=variable: self._emit("visibility", layer=layer, enabled=var.get()))
            button.grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 9), pady=2)
            self.layer_buttons[name] = button
        cameras = ttk.Frame(display)
        cameras.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(9, 0))
        self.camera_buttons = {}
        for preset, label in CAMERA_LABELS.items():
            button = ttk.Button(cameras, text=label,
                command=lambda value=preset: self._emit("camera", preset=value))
            button.pack(side="left", expand=True, fill="x", padx=2)
            self.camera_buttons[preset] = button
        self.status_var = tk.StringVar()
        ttk.Label(outer, textvariable=self.status_var, style="Muted.TLabel", wraplength=470).pack(anchor="w")
        ttk.Label(outer, text="也可在 MuJoCo 窗口用空格暂停，N/P 切换，+/- 调速。\n拖动进度定位后暂停；点击继续播放。",
                  style="Muted.TLabel", wraplength=470).pack(anchor="w", pady=(7, 0))
        self._filter_choices(publish=False)
        self._suspend = False
        self.poll(self._state)

    def _emit(self, action, **values):
        if not self._suspend and not self.closed:
            self.enqueue({"action": action, **values})

    def _scroll_wheel(self, event):
        # A drop-down owns its wheel while open; scrolling the panel must not
        # silently change the selected sample/model/speed.
        if event.widget.winfo_class() in ("TCombobox", "Listbox"):
            return
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        direction = -1 if number == 4 or delta > 0 else 1
        if number in (4, 5) or delta:
            self.canvas.yview_scroll(direction * 3, "units")
            return "break"

    def _bounce_gap_selected(self, event=None):
        index = self.bounce_gap_box.current()
        if 0 <= index < len(BOUNCE_GAPS):
            self._emit("bounce_gap", seconds=BOUNCE_GAPS[index])

    def _clear_search(self):
        self.search_var.set("")
        self._filter_choices()

    def _filter_choices(self, publish=True):
        split = next((key for key, value in SPLIT_LABELS.items() if value == self.split_var.get()), "all")
        self._choices = trajectory_choices(self.traces, split, self.search_var.get())
        self.trace_box.configure(values=[item["label"] for item in self._choices])
        current = str(self._state.get("trajectory_id", ""))
        selected = next((item for item in self._choices if item["trajectory_id"] == current), None)
        self.trace_var.set(selected["label"] if selected else "")
        self.count_var.set(f"{len(self._choices)} / {len(self.traces)} 条")
        state = "normal" if self._choices else "disabled"
        self.previous_button.configure(state=state)
        self.next_button.configure(state=state)
        if publish:
            self._emit("trajectory_pool", trajectory_ids=[item["trajectory_id"] for item in self._choices])

    def _choose(self):
        index = self.trace_box.current()
        if 0 <= index < len(self._choices):
            self._emit("select", trajectory_id=self._choices[index]["trajectory_id"])

    def _move(self, direction):
        if not self._choices:
            return
        identity = self._state.get("trajectory_id")
        found = next((i for i, item in enumerate(self._choices) if item["trajectory_id"] == identity), None)
        index = (found + direction) % len(self._choices) if found is not None else (0 if direction > 0 else -1)
        self._emit("select", trajectory_id=self._choices[index]["trajectory_id"])

    def _pause(self):
        self._emit("pause", paused=not self._state.get("paused", False))

    def _set_speed(self, value):
        value = clamp_speed(value)
        self._suspend = True
        self.speed_var.set(value)
        self.speed_label.set(f"{value:.2f}×")
        self._suspend = False
        self._emit("speed", value=value)

    def _speed_changed(self, value):
        speed = round(clamp_speed(value), 2)
        self.speed_label.set(f"{speed:.2f}×")
        self._emit("speed", value=speed)

    def _speed_release(self, event=None):
        self._speed_dragging = False
        self._set_speed(self.speed_var.get())

    def _seek_press(self, event=None):
        self._seek_dragging = True

    def _progress_preview(self, value):
        if self._seek_dragging:
            duration = float(self._state.get("duration_s", 0.))
            self.time_var.set(f"{float(value)*duration:.3f} / {duration:.3f} 秒")

    def _seek_release(self, event=None):
        self._seek_dragging = False
        self._emit("seek", fraction=min(1., max(0., self.progress_var.get())))

    def poll(self, state=None):
        """Process a small event batch and update controls from a state snapshot."""
        if self.closed:
            return False
        if state is not None:
            self._state = dict(state)
        try:
            self._suspend = True
            snapshot = self._state
            view_mode = snapshot.get("view_mode", "online")
            bounce_mode = view_mode == "bounce"
            self.view_var.set(VIEW_LABELS.get(view_mode, VIEW_LABELS["online"]))
            self.bounce_model_var.set(BOUNCE_MODEL_LABELS.get(snapshot.get("bounce_model", "gp"),
                                                           BOUNCE_MODEL_LABELS["gp"]))
            self.bounce_gap_var.set(f"{round(float(snapshot.get('bounce_gap_s', 0.))*1000)} ms")
            self.bounce_compare_var.set(bool(snapshot.get("bounce_compare", True)))
            for widget in (self.bounce_model_box, self.bounce_gap_box):
                widget.configure(state="readonly" if bounce_mode else "disabled")
            self.bounce_compare_button.configure(state="normal" if bounce_mode else "disabled")
            self.view_help_var.set("输入在触地前冻结；反弹后测量仅用于参考。紫色为固定模型对照，预测不再更新。"
                                   if bounce_mode else
                                   "在线模式持续接收观测并更新预测。切换上方观察内容，可比较反弹模型。")
            self.bounce_status_var.set(str(snapshot.get("bounce_status", "")) if bounce_mode else "")
            identity = str(snapshot.get("trajectory_id", ""))
            item = next((entry for entry in self._choices if entry["trajectory_id"] == identity), None)
            self.trace_var.set(item["label"] if item else "")
            self.identity_var.set(identity)
            self.pause_button.configure(text="继续播放" if snapshot.get("paused") else "暂停")
            self.loop_var.set(bool(snapshot.get("loop", False)))
            self.auto_next_var.set(bool(snapshot.get("auto_next", False)))
            if not self._speed_dragging:
                speed = clamp_speed(snapshot.get("speed", .5))
                self.speed_var.set(speed)
                self.speed_label.set(f"{speed:.2f}×")
            elapsed, duration = float(snapshot.get("elapsed_s", 0.)), float(snapshot.get("duration_s", 0.))
            if not self._seek_dragging:
                self.progress_var.set(0. if duration <= 0 else min(1., max(0., elapsed/duration)))
                self.time_var.set(f"{elapsed:.3f} / {duration:.3f} 秒")
            for layer, variable in self.layer_vars.items():
                variable.set(bool(snapshot.get("visible", {}).get(layer, True)))
                self.layer_buttons[layer].configure(text=(BOUNCE_LAYER_LABELS if bounce_mode else LAYER_LABELS)[layer])
            self.status_var.set(f"已收到 {snapshot.get('raw_rows_received', 0)} 个观测 · "
                                f"{'冻结预测已计算' if bounce_mode else '预测已更新'} "
                                f"{snapshot.get('forecast_updates', 0)} 次")
            self._suspend = False
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True
            return False
        finally:
            self._suspend = False
        return not self.closed

    def close(self):
        if self.closed:
            return
        self._emit("panel_close")
        self.closed = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass
