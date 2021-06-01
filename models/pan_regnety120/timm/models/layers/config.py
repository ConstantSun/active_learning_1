""" Model / Layer Config singleton state
"""
from typing import Any, Optional

__all__ = [
    'is_exportable', 'is_scriptable', 'is_no_jit',
    'set_exportable', 'set_scriptable', 'set_no_jit', 'set_layer_config'
]

# Set to True if prefer to have layers with no jit optimization (includes activations)
_NO_JIT = False

# Set to True if prefer to have activation layers with no jit optimization
# NOTE not currently used as no difference between no_jit and no_activation jit as only layers obeying
# the jit flags so far are activations. This will change as more layers are updated and/or added.
_NO_ACTIVATION_JIT = False

# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

# Set to True if wanting to use torch.jit.script on a model
_SCRIPTABLE = False


def is_no_jit():
    """
    Returns true if is is_NO_J

    Args:
    """
    return _NO_JIT


class set_no_jit:
    def __init__(self, mode: bool) -> None:
        """
        Initialize global global j - link.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
        """
        global _NO_JIT
        self.prev = _NO_JIT
        _NO_JIT = mode

    def __enter__(self) -> None:
        """
        Enter the callable call.

        Args:
            self: (todo): write your description
        """
        pass

    def __exit__(self, *args: Any) -> bool:
        """
        Execute the exit.

        Args:
            self: (todo): write your description
        """
        global _NO_JIT
        _NO_JIT = self.prev
        return False


def is_exportable():
    """
    Return the exportable of the exportable function?

    Args:
    """
    return _EXPORTABLE


class set_exportable:
    def __init__(self, mode: bool) -> None:
        """
        Initialize the global mode.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
        """
        global _EXPORTABLE
        self.prev = _EXPORTABLE
        _EXPORTABLE = mode

    def __enter__(self) -> None:
        """
        Enter the callable call.

        Args:
            self: (todo): write your description
        """
        pass

    def __exit__(self, *args: Any) -> bool:
        """
        Exit the exit code.

        Args:
            self: (todo): write your description
        """
        global _EXPORTABLE
        _EXPORTABLE = self.prev
        return False


def is_scriptable():
    """
    Returns true if the script is a script.

    Args:
    """
    return _SCRIPTABLE


class set_scriptable:
    def __init__(self, mode: bool) -> None:
        """
        Initialize the global mode.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
        """
        global _SCRIPTABLE
        self.prev = _SCRIPTABLE
        _SCRIPTABLE = mode

    def __enter__(self) -> None:
        """
        Enter the callable call.

        Args:
            self: (todo): write your description
        """
        pass

    def __exit__(self, *args: Any) -> bool:
        """
        Exit the exit.

        Args:
            self: (todo): write your description
        """
        global _SCRIPTABLE
        _SCRIPTABLE = self.prev
        return False


class set_layer_config:
    """ Layer config context manager that allows setting all layer config flags at once.
    If a flag arg is None, it will not change the current value.
    """
    def __init__(
            self,
            scriptable: Optional[bool] = None,
            exportable: Optional[bool] = None,
            no_jit: Optional[bool] = None,
            no_activation_jit: Optional[bool] = None):
        """
        Initialize the script.

        Args:
            self: (todo): write your description
            scriptable: (str): write your description
            exportable: (str): write your description
            no_jit: (bool): write your description
            no_activation_jit: (todo): write your description
        """
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        self.prev = _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        if scriptable is not None:
            _SCRIPTABLE = scriptable
        if exportable is not None:
            _EXPORTABLE = exportable
        if no_jit is not None:
            _NO_JIT = no_jit
        if no_activation_jit is not None:
            _NO_ACTIVATION_JIT = no_activation_jit

    def __enter__(self) -> None:
        """
        Enter the callable call.

        Args:
            self: (todo): write your description
        """
        pass

    def __exit__(self, *args: Any) -> bool:
        """
        Exit the exit code.

        Args:
            self: (todo): write your description
        """
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT = self.prev
        return False
