"""A scalar automatic-differentiation engine (steps 1-3).

Step 1 - the Value node stores
    data      : the scalar value computed in the forward pass
    grad      : dL/d(this node), filled in by backward(), starts at 0
    _parents  : the nodes this value was computed from (the edges of the graph)
    _op       : the operation that produced it ('+', '*', '**k', 'relu', ...)
    _backward : the LOCAL derivative rule - a closure that takes out.grad and adds
                (local derivative x out.grad) to each parent's grad (chain rule)

Step 2 - every operator below returns a NEW Value whose parents are its inputs, so evaluating an
expression builds the computation graph as a side effect of the forward pass.
    add   c = a + b    dc/da = 1            dc/db = 1
    mul   c = a * b    dc/da = b            dc/db = a
    pow   c = a ** k   dc/da = k * a**(k-1)            (k is a constant int/float)
    relu  c = max(0,a) dc/da = 1 if a > 0 else 0
Extra operations used to train the model in step 5 (built on the same mechanism):
    exp   c = e**a     dc/da = e**a
    log   c = ln(a)    dc/da = 1 / a
    sigmoid c = 1/(1+e**-a)  dc/da = c * (1 - c)
    neg, sub and div are expressed with mul, add and pow.

Step 3 - backward(): topological sort of the graph, output grad = 1, visit nodes in reverse
topological order and ACCUMULATE (+=) into the parents' grads, so a node used several times
receives the sum of all its contributions.
"""

from __future__ import annotations

import math


class Value:
    __slots__ = ("data", "grad", "_parents", "_op", "_backward", "label")

    def __init__(self, data: float, _parents: tuple = (), _op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(_parents)
        self._op = _op
        self._backward = lambda: None  # leaf nodes have no local rule
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"

    @staticmethod
    def _wrap(x) -> "Value":
        return x if isinstance(x, Value) else Value(x)

    # ---- step 2: the four required operations ----------------------------------------------
    def __add__(self, other) -> "Value":
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = Value._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k) -> "Value":
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        out = Value(self.data ** k, (self,), f"**{k:g}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---- extra operations for the training example --------------------------------------------
    def exp(self) -> "Value":
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Value._wrap(other))

    def __truediv__(self, other):
        return self * Value._wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value._wrap(other) - self

    def __rtruediv__(self, other):
        return Value._wrap(other) / self

    # ---- step 3: backward ---------------------------------------------------------------------
    def topological_order(self) -> list["Value"]:
        """Every node after all of its parents (iterative DFS - no recursion limit on deep graphs)."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        return order

    def backward(self) -> list["Value"]:
        order = self.topological_order()   # 1. topological sort
        self.grad = 1.0                    # 2. d(out)/d(out) = 1
        for node in reversed(order):       # 3. reverse order: a node's grad is complete before use
            node._backward()               # 4. each rule ADDS into its parents' grads
        return order


def total(values) -> Value:
    """sum() for Values (a chain of + nodes)."""
    out = Value(0.0)
    for v in values:
        out = out + v
    return out


def zero_grad(nodes) -> None:
    for n in nodes:
        n.grad = 0.0


def graph_summary(root: Value) -> dict:
    order = root.topological_order()
    ops = {}
    for n in order:
        ops[n._op or "leaf"] = ops.get(n._op or "leaf", 0) + 1
    return {"nodes": len(order), "edges": sum(len(n._parents) for n in order), "ops": ops}


def to_dot(root: Value) -> str:
    """Graphviz DOT text of the graph (render with any Graphviz viewer)."""
    order = root.topological_order()
    ids = {id(n): f"n{i}" for i, n in enumerate(order)}
    lines = ["digraph G {", "  rankdir=LR;", "  node [shape=record, fontsize=10];"]
    for n in order:
        name = n.label or (n._op or "leaf")
        lines.append(f'  {ids[id(n)]} [label="{{ {name} | data {n.data:.4f} | grad {n.grad:.4f} }}"];')
        for p in n._parents:
            lines.append(f"  {ids[id(p)]} -> {ids[id(n)]};")
    lines.append("}")
    return "\n".join(lines)
