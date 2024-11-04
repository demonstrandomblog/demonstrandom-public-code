class ENode:
    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __eq__(self, other):
        ops_equal = self.op == other.op 
        args_equal = self.args == other.args
        return ops_equal and args_equal

    def __hash__(self):
        return hash((self.op, self.args))

    def __repr__(self):
        return f"ENode({self.op}, {self.args})"


class EClass:
    def __init__(self, id_):
        self.id = id_
        self.nodes = set()

    def __repr__(self):
        return f"EClass({self.id}, {self.nodes})"


class EGraph:

    def __init__(self):
        self.classes = {}
        self.parents = {}
        self.enode_to_eclass_id = {}
        self.next_id = 0

    def _get_next_eclass_id(self):
        while True:
            yield self.next_id
            self.next_id += 1

    def add(self, enode):
        if enode in self.enode_to_eclass_id:
            return self.find(self.enode_to_eclass_id[enode])
    
        # Allocate a new id
        eclass_id = self._get_next_eclass_id()

        # Create a new eclass 
        self.classes[eclass_id] = EClass(eclass_id)
        self.classes[eclass_id].nodes.add(enode)
        self.enode_to_eclass_id[enode] = eclass_id
        self.parents[eclass_id] = eclass_id 
    
        # Skip merging for constant nodes
        if not enode.args:
            return eclass_id
        
        # Only union with congruent nodes
        for other_node in list(self.enode_to_eclass_id.keys()):
            ops_equal = other_node.op == enode.op
            args_equal_len = len(other_node.args) == len(enode.args)
            
            if ops_equal and args_equal_len:

                for arg1, arg2 in zip(other_node.args, enode.args):
                    arg1_canonical = self.find(arg1)
                    arg2_canonical = self.find(arg2)
                    if arg1_canonical != arg2_canonical:
                        break # If any arguments don't match, stop

                else: # Runs only if all arguments match
                    other_class = self.enode_to_eclass_id[other_node]
                    other_canonical_id = self.find(other_class)
                    union = self.union(eclass_id, other_canonical_id)
                    return union
            
        return eclass_id

    def find(self, id_):
        if id_ not in self.parents:  
            self.parents[id_] = id_
        if self.parents[id_] != id_:
            self.parents[id_] = self.find(self.parents[id_])
        return self.parents[id_]

    # union by size of e-class 
    def _compare_eclass_ranks(self, rep1, rep2):
        rank1 = len(self.classes[rep1].nodes)
        rank2 = len(self.classes[rep2].nodes)
        if rank2 > rank1:
            return rep2, rep1
        else: 
            # root1 wins ties
            return rep1, rep2 

    def union(self, id1, id2):
        rep1, rep2 = self.find(id1), self.find(id2)

        if rep1 == rep2:
            # No need to merge if they're the same eclass
            return rep1
        
        ranked_reps = self._compare_eclass_ranks(rep1, rep2)
        parent_rep, child_rep = ranked_reps

        # Update the child rep's parents
        self.parents[child_rep] = parent_rep

        # For each node in the child rep, update its parents
        child_nodes = self.classes[child_rep].nodes
        self.classes[parent_rep].nodes.update(child_nodes)
        for node in self.classes[child_rep].nodes:
            self.enode_to_eclass_id[node] = parent_rep

        # Delete the child eclass, since it's now merged
        del self.classes[child_rep]

        return parent_rep

    def rebuild(self):

        # Maintain a queue of nodes to be processed
        pending_nodes = list(self.enode_to_eclass_id.items())
    
        while pending_nodes:
            enode, initial_id = pending_nodes.pop(0)
            current_id = self.find(initial_id)
            new_args = tuple(self.find(arg) for arg in enode.args)
        
            if new_args != enode.args:
                new_enode = ENode(enode.op, new_args)

                # Remove the old enode
                self.classes[current_id].nodes.remove(enode)
                del self.enode_to_eclass_id[enode]

                # Add the new enode
                new_id = self.add(new_enode)

                # Merge  
                if self.find(current_id) != self.find(new_id):
                    self.union(current_id, new_id)

                # This is a fixpoint operation
                # We will need to check the new node again
                # Add it to the end of the queue
                pending_nodes.append((new_enode, new_id))

    def _rank_enodes(self, enode):
        # rank by size, then by lexical order
        # smallest number of args wins, then first alphabetically
        return (len(enode.args), enode.op)

    def extract(self, id_) :
        root = self.find(id_)
        eclass = self.classes[root]
        best_node = min(eclass.nodes, key=self._rank_enodes)
        if not best_node.args:
            return best_node.op
        canon_args = [self.extract(arg) for arg in best_node.args]
        return (best_node.op,) + tuple(canon_args)


def test_egraph_arithmetic():
    
    egraph = EGraph()

    # Add some expressions
    var = lambda name: egraph.add(ENode(name, ()))
    const = lambda x: egraph.add(ENode(str(x), ()))
    plus = lambda x, y: egraph.add(ENode('+', (x, y)))

    var_x, var_y = var('x'), var('y')
    one, two, three = const(1), const(2), const(3)
    expr1 = plus(one, two)  # 1 + 2
    expr2 = plus(two, one)  # 2 + 1 - note that this is never unioned 

    egraph.union(var_x, one) # Set x = 1
    egraph.union(var_y, two) # Set y = 2
    egraph.union(expr1, three) # 1 + 2 == 3
    egraph.union(plus(var_x, var_y), plus(var_y, var_x))

    # Rebuild to propagate changes
    egraph.rebuild()

    # Since we know x + y == y + x, we can conclude 1 + 2 == 2 + 1
    assert egraph.extract(expr1) == egraph.extract(expr2)

    # Since we know 1 + 2 == 3, and we know commutativity, 2 + 1 == 3
    assert egraph.extract(expr2) == egraph.extract(three)


def test_egraph_multiplication_optimization():
    egraph = EGraph()

    mul = lambda a, b: egraph.add(ENode('*', (a, b)))
    var = lambda name: egraph.add(ENode(name, ()))
    
    x = var('x')
    y = var('y')    
    
    c = var('c')
    expr1 = mul(x, y)
    egraph.union(c, expr1) 
    
    expr2 = mul(mul(x, y), mul(x, y))

    egraph.rebuild()

    print(egraph.extract(expr2)) # prints (*, 'c', 'c')
    assert egraph.extract(expr2) == egraph.extract(mul(c, c))

    return egraph 


if __name__ == "__main__":
    test_egraph_arithmetic()
    test_egraph_multiplication_optimization()