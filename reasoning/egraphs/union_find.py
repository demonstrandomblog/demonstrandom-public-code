class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            # Path compression
            self.parent[x] = self.find(self.parent[x]) 
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def test_unionfind():
    uf = UnionFind()

    # Adding elements
    for char in "abcdexy":
        uf.make_set(char)

    # Performing unions
    uf.union("x", "a")
    uf.union("y", "b")
    uf.union("a", "b")
    uf.union("c", "d")

    # Test 
    assert "x" == uf.find("a") 
    assert "x" == uf.find("b")
    assert "e" == uf.find("e")
    assert uf.find("x") == uf.find("y") # checks x ?= y

if __name__ == "__main__":
    test_unionfind()
