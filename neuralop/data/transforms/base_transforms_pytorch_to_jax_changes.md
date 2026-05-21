# base_transforms.py → base_transforms_jax.py: Changes and Rationale

## 1. Imports

| PyTorch | JAX |
|---|---|
| `import torch` | `import jax`, `import jax.numpy as jnp` |
| `torch.nn.Module` (base class) | `from flax import linen as nn` → `nn.Module` |

**Why:** JAX has no built-in neural network module system. Flax's `linen` (`nn.Module`) is the standard replacement for `torch.nn.Module`. It provides a structured, functional-style module system compatible with JAX's stateless paradigm.

---

## 2. `Transform` base class

### Removed: `__init__` with `super().__init__()`
Flax modules are dataclasses; initialization is handled automatically. There is no need to call `super().__init__()` manually.

### Removed: `cuda()` and `cpu()` abstract methods
JAX does not have `.cuda()` / `.cpu()` methods on modules or arrays. Device placement is done explicitly via `jax.device_put(array, device)` on parameter pytrees (dicts), not on module objects. These methods are irrelevant in the JAX paradigm.

### Changed: `to(device)` 
Kept as a no-op placeholder with a docstring explaining that JAX uses `jax.device_put` on parameter dicts, not on modules. Subclasses can override if needed.

### Added: `__call__` with `@nn.compact`
Flax modules require a `__call__` method as the entry point. Added a default that delegates to `transform()` to match Flax conventions.

---

## 3. `CompositeTransform`

### Changed: `__init__` → `transforms: List` class attribute
Flax modules declare fields as class-level type-annotated attributes (dataclass style), not in `__init__`. `transforms: List` replaces `self.transforms = transforms` inside `__init__`.

### Fixed bug: `super.__init__()` → removed (was missing parentheses in original)
The original PyTorch code had `super.__init__()` (missing `()` after `super`), which would raise a `TypeError`. In the JAX version this is moot since Flax handles init automatically.

### Fixed bug: `self.data_dict` → `data_dict` in `transform` and `inverse_transform`
The original used `self.data_dict` (an undefined attribute) instead of the local variable `data_dict`. Fixed in both `transform` and `inverse_transform`.

### Fixed bug: `inverse_transform` called `tform.transform` instead of `tform.inverse_transform`
The original `inverse_transform` loop called `tform.transform(...)` — this is clearly a bug; it should call `tform.inverse_transform(...)`. Fixed.

---

## 4. `DictTransform`

### Changed: `__init__` → Flax dataclass fields + `setup()`
Fields (`transforms`, `input_mappings`, `return_mappings`) are declared as class-level attributes. Validation logic (the `assert` statements) moved into `setup()`, which is Flax's lifecycle hook called after the module is initialized — equivalent to the body of `__init__` after field assignment.

### Changed: `torch.zeros_like` → `jnp.zeros_like`
Direct substitution; `jnp.zeros_like` is the JAX/NumPy equivalent.

### Changed: `out[indices] = encoded` → `out = out.at[indices].set(encoded)`
JAX arrays are **immutable**. In-place mutation (`out[indices] = ...`) is not allowed. JAX provides the `.at[...].set(...)` indexed update syntax which returns a new array with the update applied. This is a fundamental JAX paradigm shift from PyTorch.

### Changed: `self.return_mappings` truthiness check → `is not None`
Using `if self.return_mappings:` would be falsy for an empty dict. Changed to `if self.return_mappings is not None:` for correctness.

### Removed: `cpu()` and `cuda()` methods
Same reason as in `Transform` base class — not applicable in JAX. The `to()` method is kept as a no-op placeholder.

### Changed: `self.encoders = ...` in `cpu/cuda/to` → removed / `to` is no-op
The original PyTorch methods wrote to `self.encoders` (which doesn't exist — likely a bug, should be `self.transforms`). In JAX, device placement is external to the module, so these are replaced with a single `to()` no-op.

---

## Summary Table

| Aspect | PyTorch | JAX/Flax |
|---|---|---|
| Base class | `torch.nn.Module` | `flax.linen.nn.Module` |
| Module init | `__init__` + `super().__init__()` | Dataclass fields + `setup()` |
| Device move | `.cuda()`, `.cpu()`, `.to(device)` | `jax.device_put(params, device)` |
| Array zeros | `torch.zeros_like(x)` | `jnp.zeros_like(x)` |
| In-place update | `out[idx] = val` | `out = out.at[idx].set(val)` |
| Entry point | `forward()` (implicit) | `__call__()` (explicit, `@nn.compact`) |
| Tensor type | `torch.Tensor` | `jnp.ndarray` |
