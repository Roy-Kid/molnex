---
name: molnex-perf
description: Performance profiling and PyTorch optimization review. Use for performance-critical code like encoders, potentials, and message passing.
argument-hint: "[path or module]"
user-invocable: true
---

Review performance for: $ARGUMENTS

If no path given, check all files modified in `git diff --name-only HEAD`.

**Checks**

1. **torch.compile compatibility**
   - Dynamic control flow (data-dependent if/for) → graph breaks
   - Python-side ops in hot paths (list comprehensions over tensors)
   - `torch.autograd.Function` without `setup_context` (torch>=2.6)
   - `.item()` calls inside forward pass

2. **Memory anti-patterns**
   - Unnecessary `.detach().clone()` chains
   - Missing `del` for large intermediates in long forward passes
   - Gradient accumulation without `torch.no_grad()` in eval
   - Large tensor materialization that could be lazy

3. **GPU utilization**
   - CPU-GPU transfers in hot paths (`.cpu()`, `.numpy()` inside forward)
   - Unnecessary `torch.cuda.synchronize()`
   - Small kernel launches on tiny tensors
   - Missing `pin_memory=True` in DataLoader

4. **Numerical stability**
   - Division without epsilon guard
   - `log` without input clamping
   - Float32/64 consistency with `molix.config.ftype`

5. **cuEquivariance optimization**
   - Tensor products must use cuEquivariance, not manual einsum
   - `layout=cue.ir_mul` used consistently
   - Batch dimensions handled correctly

6. **Modern PyTorch (>=2.6)**
   - Deprecated APIs: `torch.cuda.amp.autocast` → `torch.autocast`
   - `torch.compile` readiness
   - `torch.export` compatibility

**Output format**:
```
PERFORMANCE REVIEW: <path>

✅ <check passed>
⚠️ [SEVERITY] Line N: <issue> — <recommendation>
❌ [SEVERITY] Line N: <issue> — <recommendation>

N ERRORS, M WARNINGS
```
