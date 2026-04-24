---
name: molnex-docs
description: Documentation and docstring completeness audit. Use before PR submission or after implementing a feature.
argument-hint: "[path or module]"
user-invocable: true
---

Audit documentation for: $ARGUMENTS

If no path given, check all files modified in `git diff --name-only HEAD`.

**Checks**

1. **Docstring presence**: Every public function, class, method in `src/` must have a docstring.

2. **Google-style format**:
   ```python
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Brief description.

       Args:
           x: Description ``(shape)``.

       Returns:
           Description ``(shape)``.
       """
   ```

3. **Tensor shape annotations**: All tensor params and returns use ``(n_nodes, dim)`` notation.

4. **Scientific reference**: Modules implementing published methods must include:
   ```
   Reference:
       Author et al. "Title" Venue Year
       https://arxiv.org/abs/XXXX.XXXXX
   ```

5. **Pydantic config docs**: All `BaseModel` configs must have class docstring with `Attributes:` section.

**Output format**:
```
DOCUMENTATION AUDIT: <path>

✅ <file>: N/N public symbols documented
⚠️ <file>: <function> missing tensor shapes (line N)
❌ <file>: No Reference section (implements published method)

Coverage: N/M (XX%)
```
