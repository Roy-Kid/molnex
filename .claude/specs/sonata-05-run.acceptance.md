---
slug: sonata-05-run
criteria:
  - id: ac-001
    summary: train.py 完成 1-epoch smoke 并落齐 metrics.json / TensorBoard / Zarr journal / last.pt
    type: runtime
    pass_when: |
      `python -u work/sonata_bulk_water/train.py --data-root <small fixture>
      --out-dir <tmp> --max-epochs 1 --batch-size 2 --lr 1e-3 --seed 0`
      退出码 0；<tmp>/metrics.json 存在且键集为
      {energy_mae_meV_per_atom, force_rmse_meV_per_A}；
      <tmp>/tb/ 至少含 1 个 events.out.tfevents.* 文件；
      <tmp>/journal/metrics/records/train/ 目录存在（Zarr v3 layout，
      JournalWriter 输出）；
      <tmp>/checkpoints/last.pt 存在且 torch.load 成功。
    status: verified
    last_checked: 2026-05-11
  - id: ac-002
    summary: NaNStopHook 在 loss=NaN 时以退出码 2 终止 + 落 nan_checkpoint.pt
    type: runtime
    pass_when: |
      临时 --debug-inject-nan flag 在 step 2 把 loss 设为 float("nan")；
      train.py 进程退出码为 2；
      <out_dir>/nan_checkpoint.pt 存在且 torch.load 成功；
      <out_dir>/train.log 末段含子串 "NaN detected"。
    status: verified
    last_checked: 2026-05-11
  - id: ac-003
    summary: submit.py --dry-run 渲染合法的 Alvis A100 sbatch
    type: runtime
    pass_when: |
      `python work/sonata_bulk_water/submit.py --account TEST-X-Y
      --time 00:30:00 --data-root /tmp/x --out-dir /tmp/y --dry-run`
      退出码 0；stdout 同时含子串 "#SBATCH -p alvis"、
      "#SBATCH --gpus-per-node=A100:1"、"#SBATCH -t 00:30:00"、
      "#SBATCH -A TEST-X-Y"；不调用 sbatch 二进制。
    status: verified
    last_checked: 2026-05-11
  - id: ac-004
    summary: submit.py 真投递返回 jobid 并写 jobid.txt
    type: runtime
    evaluator_hint: alvis-login-node
    pass_when: |
      在 Alvis 登录节点 `python work/sonata_bulk_water/submit.py --account <real>
      --time 00:30:00 --data-root <real> --out-dir <real> --max-epochs 2
      --batch-size 2` 退出码 0；<out_dir>/submit.sbatch 存在；
      <out_dir>/jobid.txt 存在且内容为 int 字符串；
      `squeue -j $(cat <out_dir>/jobid.txt) -h` 至少返回一行。
    status: pending
  - id: ac-005
    summary: 论文超参在 train.py 中显式可 grep
    type: code
    pass_when: |
      `grep -nE "^R_MAX\\s*=\\s*5\\.0|^NUM_FEATURES\\s*=\\s*64|^NUM_LAYERS\\s*=\\s*2|^SIGMA\\s*=\\s*1\\.0|^DL\\s*=\\s*2\\.0"
      work/sonata_bulk_water/train.py` 匹配 ≥ 5 行；
      README 中明确归属 Cheng B. 2025 *npj Comput. Mater.* 11:80
      (doi:10.1038/s41524-025-01577-7) 与 bm_sonata.py:51-95。
    status: verified
    last_checked: 2026-05-11
  - id: ac-006
    summary: 旧 bench/hpc spec 被删除，新 merged spec 立项
    type: code
    pass_when: |
      `.claude/specs/sonata-05-bench.md`、
      `.claude/specs/sonata-05-bench.acceptance.md`、
      `.claude/specs/sonata-05-hpc.md`、
      `.claude/specs/sonata-05-hpc.acceptance.md` 均不存在；
      `.claude/specs/sonata-05-run.md` 与
      `.claude/specs/sonata-05-run.acceptance.md` 存在；
      sonata-05-run.md frontmatter 的 supersedes 字段含
      "sonata-05-bench" 与 "sonata-05-hpc"。
    status: verified
    last_checked: 2026-05-11
  - id: ac-007
    summary: 数据加载用 MmapDataset，**不**用 CachedDataset
    type: code
    pass_when: |
      `grep -nE "MmapDataset\\(" work/sonata_bulk_water/train.py` ≥ 1 行
      （`_build_split_dataset` 末尾的 `MmapDataset(packed.sink)` 包装）；
      `grep -nE "CachedDataset" work/sonata_bulk_water/train.py` 返回空
      （**绝不**用 in-RAM 的 CachedDataset，对应 feedback_dataset_base_class 规则）；
      `grep -nE "^from molix.data.dataset import MmapDataset"
      work/sonata_bulk_water/train.py` 匹配 1 行。
    status: verified
    last_checked: 2026-05-11
  - id: ac-008
    summary: TensorBoard 与 JournalHook 默认开启且写入 out_dir
    type: code
    pass_when: |
      train.py 中可 grep 到
      `TensorBoardHook(every_n_steps=10, log_dir=str(out_dir / "tb"))` 调用；
      可 grep 到 `JournalWriter(out_dir / "journal", run_id="train")` 调用，
      且其返回值作为 `JournalHook(every_n_steps=10, store=...)` 的 store kwarg；
      两者均无 `if args.tensorboard` 之类 opt-in gate（默认开启）。
    status: verified
    last_checked: 2026-05-11
---

# Acceptance criteria

`ac-001` 锁住 smoke 路径的四份关键产物（`metrics.json` / TensorBoard / Zarr journal / checkpoint），任一缺失即说明 hook 装配或 trainer 编排被破坏。Journal 落到 `<out_dir>/journal/metrics/records/train/` 是 `JournalWriter`（Zarr v3 后端）的正常布局，而**不**是 jsonl 文件。

`ac-002` 锁住 NaN 早停语义 —— 这是 PI 在本规范中明示的硬要求；退出码 2 区分"NaN 触发"与"其他 Python 异常"（退出码 1），让 SLURM `--mail-type=FAIL` 与外层 wrapper 可以分流处理。

`ac-003` / `ac-004` 锁住 SLURM 投递路径：`--dry-run` 是工程门，真投递是端到端门。`ac-004` 标 `evaluator_hint: alvis-login-node` —— 只在 Alvis 登录节点上有意义，本地开发机跳过。

`ac-005` 把论文超参溯源固化进代码：grep 命中即等于"这些值在源码里不可隐式漂移"；改一个值会立刻丢失 grep 匹配。常量名首字母大写、模块顶层赋值是 grep 锚点的设计选择。

`ac-006` 是 spec 治理门：旧 bench/hpc 必须真删，merged spec 的 supersedes frontmatter 必须真在 —— 防止幽灵 spec 与本规范并存。

`ac-007` 把 `feedback_dataset_base_class` 的硬规则（`MmapDataset` 优于 `CachedDataset`，DataModule 接受预构造 dataset）刻进 grep-based 验证；以 `MmapDataset(` 命中 + `CachedDataset` 不命中表达"用 mmap，不用 in-RAM"的设计意图。

`ac-008` 把"代码需要 tensorboard 和 log 监视状态"的 PI 要求固化为 grep-based 验证：默认开启，无 opt-in flag，最大化"开箱即用"的观察性。grep 锚点是真实 API 形态（`TensorBoardHook(every_n_steps=…, log_dir=…)` 与 `JournalWriter(path, run_id=…)`），与 `src/molix/hooks/` 当前接口一致。
