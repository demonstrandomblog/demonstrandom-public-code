# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Intern immutable hashable objects by equality, with hash-collision handling.

HashCons.cons(value) returns the stored equal object, preserving identity across
equal values. Unequal objects remain distinct even if their hashes collide.
"""
class HashCons:
    def __init__(self):
        self.store = {}

    def cons(self, obj):
        # Dictionary lookup checks equality as well as the hash.
        return self.store.setdefault(obj, obj)


def test_hashcons():
    hs = HashCons()

    tuple1 = ("x", "+", "y")
    tuple2 = tuple(["x", "+", "y"])

    # Equal values constructed as distinct runtime objects
    assert tuple1 is not tuple2

    hashed_tuple1 = hs.cons(tuple1)
    hashed_tuple2 = hs.cons(tuple2)

    # But same values
    assert hashed_tuple1 is hashed_tuple2

def test_hashcons_preserves_unequal_values_with_the_same_hash():
    hs = HashCons()
    assert hash(-1) == hash(-2)
    assert hs.cons(-1) == -1
    assert hs.cons(-2) == -2
