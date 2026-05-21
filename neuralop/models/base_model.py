import inspect
import torch
import warnings
from pathlib import Path

# Author: Jean Kossaifi

class BaseModel(torch.nn.Module):
    """Based class for all Models

    This class has two main functionalities:
    * It monitors the creation of subclass, that are automatically registered
      for users to use by name using the library's config system
    * When a new instance of this class is created, the init call is intercepted
      so we can store the parameters used to create the instance.
      This makes it possible to save trained models along with their init parameters,
      and therefore load saved models easily.

    Notes
    -----
    Model can be versioned using the _version class attribute.
    This can be used for sanity check when loading models from checkpoints to verify the
    model hasn't been updated since.
    """

    _models = dict()
    _version = "0.1.0"

    def __init_subclass__(cls, name=None, **kwargs):
        """When a subclass is created, register it in _models
        We look for an existing name attribute.
        If not give, then we use the class' name.
        """
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseModel._models[name.lower()] = cls
            cls._name = name
        else:
            # warnings.warn(f'Creating a subclass of BaseModel {cls.__name__} with no name, initializing with {cls.__name__}.')
            BaseModel._models[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        """Verify arguments and save init kwargs for loading/saving

        We inspect the class' signature and check for unused parameters, or
        parameters not passed.

        We store all the args and kwargs given so we can duplicate the instance transparently.
        """
        # print(f"\n[BaseModel.__new__] STEP 1: __new__ called for class '{cls.__name__}'")
        # print(f"[BaseModel.__new__]   args passed   : {args}")
        # print(f"[BaseModel.__new__]   kwargs passed : {list(kwargs.keys())}")

        sig = inspect.signature(cls)
        model_name = cls.__name__

        verbose = kwargs.get("verbose", False)
        # Verify that given parameters are actually arguments of the model
        unused = []
        for key in kwargs:
            if key not in sig.parameters:
                unused.append(key)
                if verbose:
                    print(
                        f"Given argument key={key} "
                        f"that is not in {model_name}'s signature."
                    )
        if unused:
            print(f"[BaseModel.__new__]   kwargs NOT in {model_name}'s signature (will be ignored): {unused}")

        # Check for model arguments not specified in the configuration
        filled_defaults = {}
        for key, value in sig.parameters.items():
            if (value.default is not inspect._empty) and (key not in kwargs):
                if verbose:
                    print(
                        f"Keyword argument {key} not specified for model {model_name}, "
                        f"using default={value.default}."
                    )
                kwargs[key] = value.default
                filled_defaults[key] = value.default
        # print(f"[BaseModel.__new__]   STEP 2: Defaults filled in for missing kwargs: {list(filled_defaults.keys())}")

        if hasattr(cls, "_version"):
            kwargs["_version"] = cls._version
            # print(f"[BaseModel.__new__]   STEP 3: Injected _version = '{cls._version}'")
        kwargs["args"] = args
        kwargs["_name"] = cls._name
        # print(f"[BaseModel.__new__]   STEP 4: Injected _name = '{cls._name}'")

        # print(f"[BaseModel.__new__]   STEP 5: Calling super().__new__(cls) → torch.nn.Module.__new__({cls.__name__})")
        instance = super().__new__(cls)

        instance._init_kwargs = kwargs
        # print(f"[BaseModel.__new__]   STEP 6: Stored _init_kwargs on instance (keys: {list(kwargs.keys())})")
        # print(f"[BaseModel.__new__]   STEP 7: Returning uninitialized instance of '{cls.__name__}'")

        return instance

    def state_dict(self, destination: dict=None, prefix: str='', keep_vars: bool=False):
        """
        state_dict subclasses nn.Module.state_dict() and adds a metadata field
        to track the model version and ensure only compatible saves are loaded.

        Parameters
        ----------
        destination : dict, optional
            If provided, the state of module will
            be updated into the dict and the same object is returned.
            Otherwise, an OrderedDict will be created and returned, by default None
        prefix : str, optional
            a prefix added to parameter and buffer
            names to compose the keys in state_dict, by default ``''``
        keep_vars (bool, optional): by default the torch.Tensors
            returned in the state dict are detached from autograd.
            If True, detaching will not be performed, by default False

        """
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if state_dict.get('_metadata') == None:
            state_dict['_metadata'] = self._init_kwargs
        else:
            warnings.warn("Attempting to update metadata for a module with metadata already in self.state_dict()")
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """load_state_dict subclasses nn.Module.load_state_dict() and adds a metadata field
        to track the model version and ensure only compatible saves are loaded.

        Parameters
        ----------
        state_dict : dict
            state dictionary generated by ``nn.Module.state_dict()``
        strict : bool, optional
            whether to strictly enforce that the keys in ``state_dict``
            match the keys returned by this module's, by default True.
        assign : bool, optional
            whether to assign items in the state dict to their corresponding keys
            in the module instead of copying them inplace into the module's current
            parameters and buffers. When False, the properties of the tensors in the
            current module are preserved while when True, the properties of the Tensors
            in the state dict are preserved, by default False

        Returns
        -------
        _type_
            _description_
        """
        metadata = state_dict.pop("_metadata", None)

        if metadata is not None:
            saved_version = metadata.get("_version", None)
            if saved_version is None:
                warnings.warn(f"Saved instance of {self.__class__} has no stored version attribute.")
            if saved_version != self._version:
                warnings.warn(
                    f"Attempting to load a {self.__class__} of version {saved_version},"
                    f"But current version of {self.__class__} is {self._version}"
                )
            # remove state dict metadata at the end to ensure proper loading with PyTorch module
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def save_checkpoint(self, save_folder, save_name):
        """Saves the model state and init param in the given folder under the given name"""
        save_folder = Path(save_folder)
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        state_dict_filepath = save_folder.joinpath(f"{save_name}_state_dict.pt").as_posix()
        torch.save(self.state_dict(), state_dict_filepath)
        metadata_filepath = save_folder.joinpath(f"{save_name}_metadata.pkl").as_posix()
        # Objects (e.g. GeLU) are not serializable by json - find a better solution in the future
        torch.save(self._init_kwargs, metadata_filepath)

    def load_checkpoint(self, save_folder, save_name, map_location=None):
        save_folder = Path(save_folder)
        state_dict_filepath = save_folder.joinpath(f'{save_name}_state_dict.pt').as_posix()
        self.load_state_dict(torch.load(state_dict_filepath, map_location=map_location, weights_only=False))
    
    @classmethod
    def from_checkpoint(cls, save_folder, save_name, map_location=None):
        save_folder = Path(save_folder)

        metadata_filepath = save_folder.joinpath(f"{save_name}_metadata.pkl").as_posix()
        init_kwargs = torch.load(metadata_filepath, weights_only=False)

        version = init_kwargs.pop("_version", None)
        if hasattr(cls, "_version") and version != cls._version:
            warnings.warn(f'Checkpoint saved for version {version} of model {cls._name} but current code is version {cls._version}')
        
        # Remove metadata fields that shouldn't be passed to __init__
        init_kwargs.pop("_name", None)

        if "args" in init_kwargs:
            init_args = init_kwargs.pop("args")
        else:
            init_args = []
        instance = cls(*init_args, **init_kwargs)

        instance.load_checkpoint(save_folder, save_name, map_location=map_location)
        return instance


def available_models():
    """List the available neural operators"""
    return list(BaseModel._models.keys())


def get_model(config):
    """Returns an instantiated model for the given config

    * Reads the model to be used from config['arch']
    * Adjusts config["arch"]["data_channels"] accordingly if multi-grid patching is used

    Also prints warnings for safety, in case::
    * some given arguments aren't actually used by the model
    * some keyword arguments of the models aren't provided by the config

    Parameters
    ----------
    config : Bunch or dict-like
        configuration, must have
        arch = config['arch'] (string)
        and the corresponding config[arch] (a subdict with the kwargs of the model)

    Returns
    -------
    model : nn.Module
        the instanciated module
    """
    # print(f"\n{'='*60}")
    # print(f"[get_model] STEP 1: Reading model_arch from config")
    arch = config.model["model_arch"].lower()
    # print(f"[get_model]   arch = '{arch}'")

    model_config = config.model.copy()
    # print(f"[get_model]   raw model_config keys: {list(model_config.keys())}")

    # Remove model_arch from config as it's not a model parameter
    model_config.pop("model_arch", None)
    # print(f"[get_model]   STEP 2: Popped 'model_arch' from model_config")

    # Set the number of input channels depending on channels in data + mg patching
    data_channels = model_config.pop("data_channels")
    # print(f"[get_model]   STEP 3: Popped 'data_channels' = {data_channels}")
    try:
        patching_levels = config["patching"]["levels"]
    except KeyError:
        patching_levels = 0
    # print(f"[get_model]   patching_levels = {patching_levels}")
    if patching_levels:
        data_channels *= patching_levels + 1
        # print(f"[get_model]   data_channels scaled to {data_channels} due to patching")
    model_config["in_channels"] = data_channels
    # print(f"[get_model]   STEP 4: Set in_channels = {data_channels} in model_config")
    # print(f"[get_model]   Final model_config being passed to {arch.upper()}:")
    # for k, v in model_config.items():
        # print(f"[get_model]     {k} = {v}")

    # Dispatch model creation
    # print(f"\n[get_model]   STEP 5: Dispatching → BaseModel._models['{arch}'](**model_config)")
    # print(f"[get_model]   This calls {arch.upper()}.__new__ then {arch.upper()}.__init__")
    # print(f"{'='*60}\n")
    try:
        model = BaseModel._models[arch](**model_config)
        # print(f"\n[get_model]   STEP 6: {arch.upper()} instance fully constructed → returning to caller")
        # print(f"{'='*60}\n")
        return model
    except KeyError:
        raise ValueError(f"Got config.arch={arch}, expected one of {available_models()}.")
