
from typing import Optional, Sequence, Union, Dict

from ...utils_jax import count_tensor_params
from .base_transforms_jax import Transform, DictTransform
# import torch
import jax
import jax.numpy as jnp 

from flax import struct

@struct.dataclass
class Normalizer:
    mean: jnp.ndarray
    std: jnp.ndarray
    eps: float = 1e-6

    def transform(self, data):
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data):
        return (data * (self.std + self.eps)) + self.mean

# def _count_elements(shape: Sequence[int], axis: Optional[Sequence[int]]) -> int:
#     """Helper to replicate the original 'count_tensor_params'."""
#     if axis is None:
#         return math.prod(shape)
#     return math.prod([shape[i] for i in axis])

@struct.dataclass
class UnitGaussianNormalizer:
    """
    Normalizes data to zero mean and unit std.
    Implemented as a Flax PyTreeNode for functional purity and JAX compatibility.
    """
    # Dynamic variables (JAX arrays) that update during fitting
    mean: Optional[jnp.ndarray] = None
    std: Optional[jnp.ndarray] = None
    squared_mean: Optional[jnp.ndarray] = None
    n_elements: Union[int, jnp.ndarray] = 0
    mask: Optional[jnp.ndarray] = None
    
    # Static parameters (Excluded from JAX's state tree)
    eps: float = struct.field(pytree_node=False, default=1e-7)
    axis: Optional[Sequence[int]] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(cls, eps: float = 1e-7, axis: Optional[Union[int, Sequence[int]]] = None, mask: Optional[jnp.ndarray] = None) -> "UnitGaussianNormalizer":
        """Initialize the normalizer."""
        if isinstance(axis, int):
            axis = (axis,)
        elif axis is not None:
            axis = tuple(axis)
        return cls(eps=eps, axis=axis, mask=mask)

    def fit(self, data_batch: jnp.ndarray) -> "UnitGaussianNormalizer":
        """Returns a NEW instance of the normalizer with updated stats."""
        return self.update_mean_std(data_batch)

    def partial_fit(self, data_batch: jnp.ndarray, batch_size: int = 1) -> "UnitGaussianNormalizer":
        """Incrementally fits the normalizer and returns the updated instance."""
        if 0 in data_batch.shape:
            return self
        
        count = 0
        n_samples = len(data_batch)
        state = self  # We must carry the new state through the loop
        
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            if state.n_elements > 0:
                state = state.incremental_update_mean_std(samples)
            else:
                state = state.update_mean_std(samples)
            count += batch_size
            
        return state

    def update_mean_std(self, data_batch: jnp.ndarray) -> "UnitGaussianNormalizer":
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.axis)
            mean = jnp.mean(data_batch, axis=self.axis, keepdims=True)
            squared_mean = jnp.mean(data_batch**2, axis=self.axis, keepdims=True)
            std = jnp.std(data_batch, axis=self.axis, keepdims=True)
        else:
            batch_size = data_batch.shape[0]
            axis_no_batch = tuple(i - 1 for i in self.axis if i > 0)
            
            # Pure equivalent of: data_batch[:, self.mask == 1] = 0
            masked_data = jnp.where(self.mask == 1, 0.0, data_batch)
            
            n_elements = jnp.count_nonzero(self.mask, axis=axis_no_batch) * batch_size
            
            mean = jnp.sum(masked_data, axis=axis_no_batch, keepdims=True) / n_elements
            squared_mean = jnp.sum(masked_data**2, axis=axis_no_batch, keepdims=True) / n_elements
            std = jnp.std(data_batch, axis=self.axis, keepdims=True)
            
        # Instead of `self.mean = mean`, we return an updated copy (Functional Purity)
        return self.replace(mean=mean, std=std, squared_mean=squared_mean, n_elements=n_elements)

    def incremental_update_mean_std(self, data_batch: jnp.ndarray) -> "UnitGaussianNormalizer":
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.axis)
            axis = self.axis
            masked_data = data_batch
        else:
            axis = tuple(i - 1 for i in self.axis if i > 0)
            n_elements = jnp.count_nonzero(self.mask, axis=axis) * data_batch.shape[0]
            masked_data = jnp.where(self.mask == 1, 0.0, data_batch)

        total_elements = self.n_elements + n_elements

        mean = (1.0 / total_elements) * (
            self.n_elements * self.mean + jnp.sum(masked_data, axis=axis, keepdims=True)
        )
        squared_mean = (1.0 / total_elements) * (
            self.n_elements * self.squared_mean + jnp.sum(masked_data**2, axis=axis, keepdims=True)
        )

        # jnp.maximum is used to prevent negative values inside sqrt due to float precision limits
        variance = jnp.maximum(0.0, squared_mean - mean**2)
        std = jnp.sqrt(variance) * (total_elements / jnp.maximum(1.0, total_elements - 1))

        return self.replace(mean=mean, squared_mean=squared_mean, std=std, n_elements=total_elements)

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * (self.std + self.eps) + self.mean

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)

    @classmethod
    def from_dataset(cls, dataset, axis=None, keys=None, mask=None) -> Dict[str, "UnitGaussianNormalizer"]:
        instances = {}
        for i, data_dict in enumerate(dataset):
            if i == 0:
                if keys is None:
                    keys = data_dict.keys()
                instances = {key: cls.create(axis=axis, mask=mask) for key in keys}
            
            for key, sample in data_dict.items():
                if key in keys:
                    # sample.unsqueeze(0) -> jnp.expand_dims
                    sample_batched = jnp.expand_dims(jnp.asarray(sample), axis=0)
                    # Must re-assign to capture the updated state
                    instances[key] = instances[key].partial_fit(sample_batched)
        return instances

# Assuming UnitGaussianNormalizer is imported/defined from the previous response

@struct.dataclass
class DictUnitGaussianNormalizer:
    """
    Composes multiple UnitGaussianNormalizers to normalize different
    fields of a tensor to Gaussian distributions with mean 0 and unit variance.
    """
    # JAX natively treats dictionaries of PyTrees as PyTrees themselves!
    normalizer_dict: Dict[str, "UnitGaussianNormalizer"]
    
    # Python slice objects cannot be JAX arrays, so they MUST be marked as static fields
    input_mappings: Dict[str, slice] = struct.field(pytree_node=False)
    return_mappings: Dict[str, slice] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        normalizer_dict: Dict[str, "UnitGaussianNormalizer"],
        input_mappings: Dict[str, slice],
        return_mappings: Dict[str, slice],
    ) -> "DictUnitGaussianNormalizer":
        
        assert set(normalizer_dict.keys()) == set(input_mappings.keys()), \
            "Error: normalizers and model input fields must be keyed identically"
        assert set(normalizer_dict.keys()) == set(return_mappings.keys()), \
            "Error: normalizers and model output fields must be keyed identically"

        return cls(
            normalizer_dict=normalizer_dict,
            input_mappings=input_mappings,
            return_mappings=return_mappings,
        )

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies normalizations to specific slices of the tensor."""
        result = x
        for key, norm in self.normalizer_dict.items():
            in_slice = self.input_mappings[key]
            out_slice = self.return_mappings[key]
            
            # Extract the slice, transform it, and place it in the new tensor
            normalized_field = norm.transform(x[in_slice])
            result = result.at[out_slice].set(normalized_field)
            
        return result

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reverses the normalization on specific slices of the tensor."""
        result = x
        for key, norm in self.normalizer_dict.items():
            in_slice = self.input_mappings[key]
            out_slice = self.return_mappings[key]
            
            # Assuming return_mappings are the locations in the output tensor
            denormalized_field = norm.inverse_transform(x[out_slice])
            result = result.at[in_slice].set(denormalized_field)
            
        return result

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)

    @classmethod
    def from_dataset(
        cls, 
        dataset, 
        input_mappings: Dict[str, slice],
        return_mappings: Dict[str, slice],
        axis=None, 
        keys=None, 
        mask=None
    ) -> "DictUnitGaussianNormalizer":
        """
        [FIXED] Initializes and functionally fits the composite normalizer 
        using data from a dataset.
        """
        # Determine keys from the first batch if not provided
        first_batch = next(iter(dataset))
        if keys is None:
            keys = [k for k in first_batch.keys() if k in input_mappings]

        # 1. Initialize the individual base normalizers
        base_instances = {
            key: UnitGaussianNormalizer.create(axis=axis, mask=mask) 
            for key in keys
        }

        # 2. Iteratively and functionally update the normalizers
        for data_dict in dataset:
            for key in keys:
                if key in data_dict:
                    sample = jnp.asarray(data_dict[key])
                    sample_batched = jnp.expand_dims(sample, axis=0)
                    
                    # Functional update: Must re-assign the returned updated object
                    base_instances[key] = base_instances[key].partial_fit(sample_batched)

        # 3. Return the fully initialized composite class
        return cls.create(
            normalizer_dict=base_instances,
            input_mappings=input_mappings,
            return_mappings=return_mappings
        )