from typing import Iterable
from typing import TypeVar, Union, overload, Literal, Optional
import pkgutil

T = TypeVar("T")


def is_installed(module_name: str) -> bool:
    """Check if module is installed

    Parameters
    ----------
    module_name : str
        module name to check

    Returns
    -------
    bool
        True if module is installed, False otherwise
    """
    return any(
        _module.name == module_name for _module in pkgutil.iter_modules()
    )


if is_installed("tqdm"):
    import tqdm

    @overload
    def check_tqdm(
        __iterable: Optional[Iterable[T]],
        ipynb: Literal[True],
        silent: Literal[False],
        **kwargs
    ) -> Union[tqdm.notebook.tqdm, Iterable[T]]:
        ...

    @overload
    def check_tqdm(
        __iterable: Optional[Iterable[T]] = None,
        ipynb: Literal[False] = False,
        silent: Literal[False] = False,
        **kwargs
    ) -> Union[tqdm.tqdm, Iterable[T]]:
        ...


@overload
def check_tqdm(
    __iterable: Optional[Iterable[T]],
    ipynb: bool,
    silent: Literal[True],
    **kwargs
) -> Iterable[T]:
    ...


def check_tqdm(
    __iterable: Optional[Iterable[T]] = None,
    ipynb: bool = False,
    silent: bool = False,
    **kwargs
) -> Iterable[T]:
    if silent or not is_installed("tqdm"):
        return __iterable
    else:
        if ipynb:
            return tqdm.notebook.tqdm(__iterable, **kwargs)
        else:
            return tqdm.tqdm(__iterable, **kwargs)


if __name__ == "__main__":
    pass
