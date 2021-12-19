"""
Proving to myself that the `Union` type can be used for encapsulating _variants_
of a type and narrowing them ergonomically and statically, precluding
`TypeGuard`, which relies on a runtime function, and a series of `isinstance`
checks.

I chose to implement a Result Monad as the type-with-variants, as it is familiar
from Rust and close to something I use in production code at work.
"""

import random
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
    runtime_checkable,
)

T = TypeVar("T")
Tm = TypeVar("Tm")
X = TypeVar("X")
Xm = TypeVar("Xm")


@runtime_checkable
class Monad(Generic[T], Protocol):
    """"A monad is a monoid in the category of endofunctors. What's the problem?"

    Monads are a special kind of functor. A functor is a morphism between
    objects of 1 or more categories. A morphism is a... y'know what, just look
    up Bartosz Milewski on YouTube.

    For the average Pythonista, a Monad is a kind of container that provides
    methods for operating on the contained object(s) without pulling it out.
    Lists are monads, for example, and its monadic operations are implemented
    with `list()`, `list.extend()`, and `map()`.

    This implementation makes `Monad` a protocol for 2 reasons: it better fits
    the definition of a monad as a functor, and implementing it as an ABC caused
    a headache with Liskov Substitution in concrete implementations.
    """

    value: T

    def wrap(self, value: T) -> "Monad[T]":
        """Wrap a value in the Monadic type.

        Haskell calls this "return" in the context of monads, "pure" in the
        context of functors. It may also be called "unit." Here, it is "wrap"
        because its inverse is "unwrap."
        """
        ...

    def unwrap(self: "Monad[T]") -> T:
        """Retrieve the value from the Monad"""
        ...

    def fmap(self: "Monad[T]", f: Callable[[T], Tm]) -> "Monad[Tm]":
        """Apply a plain function to the wrapped value, returning a new Monad
        object that contains the result.

        For functions which take multiple arguments, use `functools.partial` to
        set the additional arguments ahead of time or curry the function.
        """
        ...

    def bind(self: "Monad[T]", f: Callable[[T], "Monad[Tm]"]) -> "Monad[Tm]":
        """Apply a monadic function to the wrapped value, returning the
        function's output.

        For functions which take multiple arguments, use `functools.partial` to
        set the additional arguments ahead of time or curry the function.
        """
        ...

    def flatten(self: "Monad[Monad[T]]") -> "Monad[T]":
        """Unwrap a monadic value from inside the monad."""
        ...

    def join(self: "Monad[T]", other: "Monad[T]") -> "Monad[T]":
        """Combine the value of this monad with that of another monad containing
        the same type, returning a new monad object.

        The "flatten" and "join" operations are often treated the same in
        literature on monads. The List type illustrates how these operations are
        conceptually similar, but in Python can be operationally distinct:
        ```
        list.extend <-> (self: List[T], other: Iterable[T]) -> List[T]
        # And if we take the "list" of arguments as a list in itself, this is
        # virtually equivalent.
            <~> (deeplist: List[Iterable[T]]) -> List[T]
        ```
        The second signature requires a bit of a jump, but the two operations do
        accomplish roughly the same thing: turning a list of lists into a flat
        list of values.
        """
        ...


class ResultAbc(Generic[T, X], ABC):
    """The Result Monad is used to great effect in FP-heavy languages like Rust
    and pure FP languages like Haskell. It allows abstracting away some error
    handling, resulting in cleaner code.

    There are two variants, `Ok` and `Err`. The "variant" relationship is
    captured with a Union type alias later; the purpose of this ABC is to both
    define the interface for the programmer and to capture for type-checkers how
    operations on the two types modify the contained value regardless of which
    type is actually present at runtime.
    """
    __slots__ = "value"

    value: Union[T, X]

    @abstractmethod
    def bind(
        self: "ResultAbc[T, X]", f: Callable[[T], "Result[Tm, X]"]
    ) -> "Result[Tm, X]":
        ...

    @abstractmethod
    def and_fmap(
        self: "ResultAbc[T, X]", f: Callable[[T], Tm]
    ) -> "Result[Tm, X]":
        """Apply a plain function to the wrapped value of an `Ok` object and
        return a new `Result`, or return an `Err` unmodified.

        For functions which take multiple arguments, use `functools.partial` to
        set the additional arguments ahead of time or curry the function.
        """
        ...

    @abstractmethod
    def and_bind(
        self: "ResultAbc[T, X]", f: Callable[[T], "Result[Tm, X]"]
    ) -> "Result[Tm, X]":
        """Apply a monadic function to the value of an `Ok` object, or return an
        `Err` unmodified.

        For functions which take multiple arguments, use `functools.partial` to
        set the additional arguments ahead of time or curry the function.
        """
        ...

    @abstractmethod
    def or_fmap(self: "ResultAbc[T, X]", f: Callable[[X], Xm]) -> "Result[T, Xm]":
        """Apply a plain function to the wrapped value of an `Err` object and
        return a new `Err`, or return an `Ok` unmodified.

        For functions which take multiple arguments, use `functools.partial` to
        set the additional arguments ahead of time or curry the function.
        """
        ...

    @abstractmethod
    def or_bind(
        self: "ResultAbc[T, X]", f: Callable[[X], "Result[T, Xm]"]
    ) -> "Result[T, Xm]":
        """Apply a monadic function to the value of an `Err` object, or return an
        `Ok` unmodified.

        Different from `or_fmap`, which will always return `Err` if called on an
        `Err`, `or_bind` may return `Ok` if `f` returns `Ok`. This makes it
        possible to recover (or "un-derail" in the railroad analogy) an error
        condition.

        For functions which take multiple arguments, use `functools.partial` to
        set the additional arguments ahead of time or curry the function.
        """
        ...

    def flatten(self: "ResultAbc[T, X]") -> "Result[T, X]":
        """Return a nested `Ok` or `Err` object. If the wrapped value is not
        `Ok|Err`, the return is `self`.
        """
        if isinstance(self, ResultAbc):
            return self.unwrap()  # type: ignore
        else:
            return self  # type: ignore


class Ok(ResultAbc[T, Any]):
    value: T

    def __init__(self, value: T):
        self.value = value

    def __bool__(self) -> Literal[True]:
        return True

    def wrap(self: "Ok[T]", value: T) -> "Ok[T]":
        self.value = value
        return self

    def unwrap(self: "Ok[T]") -> T:
        return self.value

    def fmap(
        self: "Ok[T]", f: Callable[[T], Tm]
    ) -> "Result[Tm, Union[X, Exception]]":
        try:
            return Ok(f(self.value))
        except Exception as e:
            return Err(e)

    and_fmap = fmap  # type: ignore

    def bind(self: "Ok[T]", f: Callable[[T], "Result[Tm, X]"]) -> "Result[Tm, X]":
        return f(self.value)

    and_bind = bind  # type: ignore

    def or_fmap(self: "Ok[T]", f: Callable[[X], Xm]) -> "Ok[T]":
        return self

    def or_bind(self: "Ok[T]", f: Callable[[X], "ResultAbc[T, Xm]"]) -> "Ok[T]":
        return self

    def join(self: "Ok[T]", other: "Ok[T]") -> "Ok[Tuple[T, T]]":
        return Ok((self.value, other.value))


class Err(ResultAbc[Any, X]):
    value: X

    def __init__(self, value: X):
        self.value = value

    def wrap(self: "Err[X]", value: X) -> "Err[X]":
        self.value = value
        return self

    def unwrap(self: "Err[X]") -> X:
        return self.value

    def fmap(self, f: Callable[..., Any]) -> "Err[X]":
        return self

    and_fmap = fmap  # type: ignore

    def bind(self: "Err[X]", f: Callable[[T], "Result[Tm, X]"]) -> "Err[X]":
        return self

    and_bind = bind  # type: ignore

    def or_bind(
        self: "Err[X]", f: Callable[[X], "Result[T, Xm]"]
    ) -> "Result[T, Xm]":
        return f(self.value)

    def join(self: "Err[X]", other: "Err[X]") -> "Err[Tuple[X, X]]":
        return Err((self.value, other.value))

    def or_fmap(self: "Err[X]", f: Callable[[X], Xm]) -> "Err[Xm]":
        try:
            return Err(f(self.value))
        except Exception as e:
            return Err(e)  # type: ignore

    def __bool__(self) -> Literal[False]:
        return False


Result = Union[Ok[T], Err[X]]
Result.__doc__ = """
This "tagged" union relies on the __bool__ implementations in the two types.
Being annotated with `Literal`, type-checkers can accurately narrow from
`Result` to either case with a simple `if/else`.
"""


def check_f(f: Callable[..., T], *a, **kw) -> Result[T, Exception]:
    """Wrap the result of a regular function with a `Result` type. This is the
    ideal starting point for an operation that is to be carried out in the
    monadic context, but is not itself monadic.

    In Haskell, this might be the "liftm" operation. It's not implemented as a
    method here for simplicity's sake.
    """
    try:
        return Ok(f(*a, **kw))
    except Exception as e:
        return Err(e)


if __name__ == "__main__":
    # Mypy infers random.choice to return the common supertype--impressive.
    # Pyright says nothing.
    # This simulates not knowing at write-time what the result might be.
    res: Result[int, Exception] = random.choice((Ok(1), Err(ZeroDivisionError())))
    reveal_type(res) if TYPE_CHECKING else None
    if res:
        # The checker should understand res is an Ok on this path...
        reveal_type(res) if TYPE_CHECKING else None
        print("Succeeded")
        res.and_fmap(lambda x: x + 2).and_fmap(float).unwrap()
    else:
        # but an Err on this one.
        reveal_type(res) if TYPE_CHECKING else None
        print("Failed")
        res.or_fmap(str).unwrap()
    # Shows an actual, if contrived, operation.
    res2 = check_f(lambda x: 5).and_fmap(lambda x: x * x).or_fmap(str)
    reveal_type(res2) if TYPE_CHECKING else None
    # Prove that I fully implemented the Monad protocol for ResultAbc.
    assert isinstance(res2, Monad)
