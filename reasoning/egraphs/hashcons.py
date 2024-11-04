class HashCons:
    def __init__(self):
        self.store = {}

    def cons(self, obj):
        hash_id = hash(obj)
        if hash_id in self.store.keys():
            return self.store[hash_id]
        else:
            self.store[hash_id] = obj
            return obj

def test_hashcons():
    hs = HashCons()

    tuple1 = ("x", "+", "y")
    tuple2 = ("x", "+", "y")

    # Different objects
    assert tuple1 is not tuple2 

    hashed_tuple1 = hs.cons(tuple1)
    hashed_tuple2 = hs.cons(tuple2)

    # But same values
    assert hashed_tuple1 is hashed_tuple2 