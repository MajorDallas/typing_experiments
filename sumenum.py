"""
This was an experiment in using the Enum metaclass as a sum type/tagged union.
The thinking was that the "tag" would be membership in the enum.

I consider this something of a failure, at least with the current implementation
of `Sum`. Still, it is interesting for showing how types can be made members of
the Enum and instances of them can even be constructed by overriding `__call__`,
as well as more ergonomic membership testing of _instances_ by overriding the
`_missing_` class method.
"""
import random
from enum import Enum


class Sum(Enum):
    @classmethod
    def _missing_(cls, value):
        for name, val in cls.__members__.items():
            if isinstance(value, val.value):
                return cls[name]
        else:
            return super()._missing_(value)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class Foo:
    pass


class Bar:
    pass


class FooOrBar(Sum):
    FOO = Foo
    BAR = Bar


def takes_foo_or_bar(fb: FooOrBar) -> FooOrBar:
    print(f"Argument's tag is {fb.name}")
    return FooOrBar(fb)


if __name__ == "__main__":
    # Prove an object is in the sum type
    c = random.choice((Foo(), Bar()))
    assert FooOrBar(c)
    # Get the "tag", if we say "tagged union" instead of "sum type," for
    # rudimentary case handling:
    if (tag := FooOrBar(c)) is FooOrBar.FOO:
        print("got a Foo")
    else:
        print("got a Bar")
    # Not that useful for static typing, since there's no relationship from the
    # variants to the sum type.
    takes_foo_or_bar(FooOrBar(c))  # Works fine
    efoo = FooOrBar.FOO()
    assert isinstance(efoo, Foo) 
    try:
        takes_foo_or_bar(efoo)  # Incompatible argument type, and an
                                # AttributeError inside the function.
        assert isinstance(efoo, FooOrBar) 
    except (AttributeError, AssertionError) as e:
        print(e)
    # FooOrBar _can_ be used as a type guard, but instances of the member types
    # are not recognized as such. Even if we define the type with `type()` in
    # the enum body, the new types' path will still be module-scoped and have no
    # nominal relationship to the enum.
