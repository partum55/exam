import os

# Point class implementation
point_py = '''# Point Class Implementation

from math import cos, sin, radians

class Point:
    """Represents a point in two-dimensional geometric coordinates"""
    
    def __init__(self, x=0, y=0):
        """Initialize the position of a new point. The x and y coordinates can be
        specified. If they are not, the point defaults to the origin."""
        self.move(x, y)
    
    def move(self, x, y):
        """Move the point to a new location in 2D space."""
        self._x = x
        self._y = y
    
    def rotate(self, beta, other_point):
        """Rotate point around other point"""
        dx = self.get_x() - other_point.get_x()
        dy = self.get_y() - other_point.get_y()
        beta = radians(beta)
        xpoint3 = dx * cos(beta) - dy * sin(beta)
        ypoint3 = dy * cos(beta) + dx * sin(beta)
        xpoint3 = xpoint3 + other_point.get_x()
        ypoint3 = ypoint3 + other_point.get_y()
        return self.move(xpoint3, ypoint3)
    
    def get_x(self):
        return self._x
    
    def get_y(self):
        return self._y
        
    def __str__(self):
        return f"Point({self._x}, {self._y})"
'''

# Church class implementation
church_py = '''# Church Class Implementation

class Church:
    """Represents a church in a database of wooden churches"""
    
    def __init__(self, name="", year=0, place="", material="wood"):
        """Initialize a new church with provided details"""
        self.name = name
        self.year = year
        self.place = place
        self.material = material
    
    def __str__(self):
        """Return a string representation of the church"""
        return f"Church: {self.name}, built in {self.year} in {self.place}"
    
    def __repr__(self):
        """Return a developer-friendly representation of the church"""
        return f"Church('{self.name}', {self.year}, '{self.place}', '{self.material}')"
    
    def get_age(self, current_year=2023):
        """Calculate the age of the church"""
        return current_year - self.year if self.year > 0 else 0
'''

# Node implementation for linked structures
node_py = '''# Node Implementation for Linked Structures

class Node:
    """Base class for nodes in linked data structures"""
    
    def __init__(self, data=None, next=None):
        """Initialize a new node with data and next reference"""
        self.data = data
        self.next = next
    
    def __str__(self):
        """Return string representation of node data"""
        return str(self.data)
'''

# Stack implementation
stack_py = '''# Stack Implementation

class Stack:
    """Implementation of a stack using a list"""
    
    def __init__(self):
        """Initialize an empty stack"""
        self.items = []
    
    def is_empty(self):
        """Check if the stack is empty"""
        return self.items == []
    
    def push(self, item):
        """Add an item to the top of the stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item from the stack"""
        if self.is_empty():
            raise ValueError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """Return the top item from the stack without removing it"""
        if self.is_empty():
            raise ValueError("Stack is empty")
        return self.items[len(self.items)-1]
    
    def size(self):
        """Return the number of items in the stack"""
        return len(self.items)
    
    def __iter__(self):
        """Return an iterator for the stack"""
        for item in reversed(self.items):
            yield item


class LinkedStack:
    """Implementation of a stack using a linked list"""
    
    def __init__(self):
        """Initialize an empty stack"""
        self._top = None
        self._size = 0
    
    def is_empty(self):
        """Check if the stack is empty"""
        return self._top is None
    
    def push(self, item):
        """Add an item to the top of the stack"""
        self._top = Node(item, self._top)
        self._size += 1
    
    def pop(self):
        """Remove and return the top item from the stack"""
        if self.is_empty():
            raise ValueError("Stack is empty")
        item = self._top.data
        self._top = self._top.next
        self._size -= 1
        return item
    
    def peek(self):
        """Return the top item from the stack without removing it"""
        if self.is_empty():
            raise ValueError("Stack is empty")
        return self._top.data
    
    def size(self):
        """Return the number of items in the stack"""
        return self._size
    
    def __iter__(self):
        """Return an iterator for the stack"""
        def _iter_helper(node):
            if node is not None:
                _iter_helper(node.next)
                yield node.data
        
        return _iter_helper(self._top)


def check_balanced_parentheses(expression):
    """Check if parentheses, brackets, and braces in an expression are balanced"""
    stack = Stack()
    opening = "([{"
    closing = ")]}"
    
    for char in expression:
        if char in opening:
            stack.push(char)
        elif char in closing:
            if stack.is_empty():
                return False
            
            top = stack.pop()
            if opening.index(top) != closing.index(char):
                return False
    
    return stack.is_empty()


def evaluate_postfix(expression):
    """Evaluate a postfix expression using a stack"""
    stack = Stack()
    tokens = expression.split()
    
    for token in tokens:
        if token in "+-*/":
            # It's an operator, pop two operands
            if stack.size() < 2:
                raise ValueError("Invalid postfix expression")
            
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                stack.push(a + b)
            elif token == '-':
                stack.push(a - b)
            elif token == '*':
                stack.push(a * b)
            elif token == '/':
                stack.push(a / b)
        else:
            # It's an operand, push onto stack
            stack.push(float(token))
    
    # The final result should be the only item on the stack
    if stack.size() != 1:
        raise ValueError("Invalid postfix expression")
    
    return stack.pop()
'''

# Queue implementation
queue_py = '''# Queue Implementation

class Queue:
    """Implementation of a queue using a list"""
    
    def __init__(self):
        """Initialize an empty queue"""
        self.items = []
    
    def is_empty(self):
        """Check if the queue is empty"""
        return len(self.items) == 0
    
    def enqueue(self, item):
        """Add an item to the back of the queue"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return the front item from the queue"""
        if self.is_empty():
            raise ValueError("Queue is empty")
        return self.items.pop(0)
    
    def peek(self):
        """Return the front item from the queue without removing it"""
        if self.is_empty():
            raise ValueError("Queue is empty")
        return self.items[0]
    
    def size(self):
        """Return the number of items in the queue"""
        return len(self.items)


class LinkedQueue:
    """Implementation of a queue using a linked list"""
    
    def __init__(self):
        """Initialize an empty queue"""
        self.front = None
        self.rear = None
        self._size = 0
    
    def is_empty(self):
        """Check if the queue is empty"""
        return self.front is None
    
    def enqueue(self, item):
        """Add an item to the back of the queue"""
        new_node = Node(item)
        if self.is_empty():
            self.front = new_node
        else:
            self.rear.next = new_node
        self.rear = new_node
        self._size += 1
    
    def dequeue(self):
        """Remove and return the front item from the queue"""
        if self.is_empty():
            raise ValueError("Queue is empty")
        
        item = self.front.data
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self._size -= 1
        return item
    
    def peek(self):
        """Return the front item from the queue without removing it"""
        if self.is_empty():
            raise ValueError("Queue is empty")
        return self.front.data
    
    def size(self):
        """Return the number of items in the queue"""
        return self._size


class ArrayQueue:
    """Implementation of a queue using a circular array"""
    
    def __init__(self, capacity=10):
        """Initialize an empty queue with the given capacity"""
        self._data = [None] * capacity
        self._size = 0
        self._front = 0
    
    def is_empty(self):
        """Check if the queue is empty"""
        return self._size == 0
    
    def enqueue(self, item):
        """Add an item to the back of the queue"""
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        avail = (self._front + self._size) % len(self._data)
        self._data[avail] = item
        self._size += 1
    
    def dequeue(self):
        """Remove and return the front item from the queue"""
        if self.is_empty():
            raise ValueError("Queue is empty")
        result = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % len(self._data)
        self._size -= 1
        return result
    
    def peek(self):
        """Return the front item from the queue without removing it"""
        if self.is_empty():
            raise ValueError("Queue is empty")
        return self._data[self._front]
    
    def size(self):
        """Return the number of items in the queue"""
        return self._size
    
    def _resize(self, capacity):
        """Resize the array to the given capacity"""
        old = self._data
        self._data = [None] * capacity
        walk = self._front
        for k in range(self._size):
            self._data[k] = old[walk]
            walk = (1 + walk) % len(old)
        self._front = 0
'''

# Deque implementation
deque_py = '''# Deque (Double-Ended Queue) Implementation

class Deque:
    """Implementation of a deque using a list"""
    
    def __init__(self):
        """Initialize an empty deque"""
        self.items = []
    
    def is_empty(self):
        """Check if the deque is empty"""
        return len(self.items) == 0
    
    def add_front(self, item):
        """Add an item to the front of the deque"""
        self.items.insert(0, item)
    
    def add_rear(self, item):
        """Add an item to the back of the deque"""
        self.items.append(item)
    
    def remove_front(self):
        """Remove and return the front item from the deque"""
        if self.is_empty():
            raise ValueError("Deque is empty")
        return self.items.pop(0)
    
    def remove_rear(self):
        """Remove and return the back item from the deque"""
        if self.is_empty():
            raise ValueError("Deque is empty")
        return self.items.pop()
    
    def peek_front(self):
        """Return the front item from the deque without removing it"""
        if self.is_empty():
            raise ValueError("Deque is empty")
        return self.items[0]
    
    def peek_rear(self):
        """Return the back item from the deque without removing it"""
        if self.is_empty():
            raise ValueError("Deque is empty")
        return self.items[-1]
    
    def size(self):
        """Return the number of items in the deque"""
        return len(self.items)
'''

# Binary Tree implementation
binary_tree_py = '''# Binary Tree Implementation

class Position:
    """Abstract position class for tree structures"""
    
    def element(self):
        """Return the element stored at this position"""
        raise NotImplementedError('must be implemented by subclass')
    
    def __eq__(self, other):
        """Return True if other is a Position representing the same location"""
        raise NotImplementedError('must be implemented by subclass')
    
    def __ne__(self, other):
        """Return True if other does not represent the same location"""
        return not (self == other)


class Tree:
    """Abstract base class representing a tree structure"""
    
    class Position(Position):
        """An abstraction representing the location of a single element"""
        
        def element(self):
            """Return the element stored at this Position"""
            raise NotImplementedError('must be implemented by subclass')
        
        def __eq__(self, other):
            """Return True if other Position represents the same location"""
            raise NotImplementedError('must be implemented by subclass')
        
        def __ne__(self, other):
            """Return True if other does not represent the same location"""
            return not (self == other)
    
    # Abstract methods
    def root(self):
        """Return Position representing the tree's root (or None if empty)"""
        raise NotImplementedError('must be implemented by subclass')
    
    def parent(self, p):
        """Return Position representing p's parent (or None if p is root)"""
        raise NotImplementedError('must be implemented by subclass')
    
    def num_children(self, p):
        """Return the number of children that Position p has"""
        raise NotImplementedError('must be implemented by subclass')
    
    def children(self, p):
        """Generate an iteration of Positions representing p's children"""
        raise NotImplementedError('must be implemented by subclass')
    
    def __len__(self):
        """Return the total number of elements in the tree"""
        raise NotImplementedError('must be implemented by subclass')
    
    # Concrete methods
    def is_root(self, p):
        """Return True if Position p represents the root of the tree"""
        return self.root() == p
    
    def is_leaf(self, p):
        """Return True if Position p does not have any children"""
        return self.num_children(p) == 0
    
    def is_empty(self):
        """Return True if the tree is empty"""
        return len(self) == 0
    
    def depth(self, p):
        """Return the number of levels separating Position p from the root"""
        if self.is_root(p):
            return 0
        else:
            return 1 + self.depth(self.parent(p))
    
    def height(self, p=None):
        """Return the height of the subtree rooted at Position p
        
        If p is None, return the height of the entire tree.
        """
        if p is None:
            p = self.root()
            if p is None:  # empty tree
                return 0
        return self._height2(p)
    
    def _height2(self, p):
        """Return the height of the subtree rooted at Position p"""
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self._height2(c) for c in self.children(p))
    
    def positions(self):
        """Generate an iteration of the tree's positions"""
        return self.preorder()  # default to preorder traversal
    
    def preorder(self):
        """Generate a preorder iteration of positions in the tree"""
        if not self.is_empty():
            for p in self._subtree_preorder(self.root()):
                yield p
    
    def _subtree_preorder(self, p):
        """Generate a preorder iteration of positions in subtree rooted at p"""
        yield p  # visit p before its subtrees
        for c in self.children(p):
            for other in self._subtree_preorder(c):
                yield other
    
    def postorder(self):
        """Generate a postorder iteration of positions in the tree"""
        if not self.is_empty():
            for p in self._subtree_postorder(self.root()):
                yield p
    
    def _subtree_postorder(self, p):
        """Generate a postorder iteration of positions in subtree rooted at p"""
        for c in self.children(p):
            for other in self._subtree_postorder(c):
                yield other
        yield p  # visit p after its subtrees
    
    def breadthfirst(self):
        """Generate a breadth-first iteration of the positions of the tree"""
        if not self.is_empty():
            # Create a FIFO queue
            from queue import Queue
            fringe = Queue()
            fringe.put(self.root())  # start with the root
            while not fringe.empty():
                p = fringe.get()  # remove from front of the queue
                yield p  # report this position
                for c in self.children(p):
                    fringe.put(c)  # add children to back of queue
    
    def __iter__(self):
        """Generate an iteration of the tree's elements"""
        for p in self.positions():
            yield p.element()


class BinaryTree(Tree):
    """Abstract base class representing a binary tree structure"""
    
    # Additional abstract methods
    def left(self, p):
        """Return a Position representing p's left child"""
        raise NotImplementedError('must be implemented by subclass')
    
    def right(self, p):
        """Return a Position representing p's right child"""
        raise NotImplementedError('must be implemented by subclass')
    
    # Concrete methods
    def sibling(self, p):
        """Return a Position representing p's sibling (or None if no sibling)"""
        parent = self.parent(p)
        if parent is None:  # p must be the root
            return None
        else:
            if p == self.left(parent):
                return self.right(parent)  # possibly None
            else:
                return self.left(parent)  # possibly None
    
    def children(self, p):
        """Generate an iteration of Positions representing p's children"""
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)
    
    def inorder(self):
        """Generate an inorder iteration of positions in the tree"""
        if not self.is_empty():
            for p in self._subtree_inorder(self.root()):
                yield p
    
    def _subtree_inorder(self, p):
        """Generate an inorder iteration of positions in subtree rooted at p"""
        if self.left(p) is not None:
            for other in self._subtree_inorder(self.left(p)):
                yield other
        yield p  # visit p between its subtrees
        if self.right(p) is not None:
            for other in self._subtree_inorder(self.right(p)):
                yield other


class LinkedBinaryTree(BinaryTree):
    """Linked representation of a binary tree structure"""
    
    class _Node:
        """Lightweight, nonpublic class for storing a node"""
        __slots__ = '_element', '_parent', '_left', '_right'
        
        def __init__(self, element, parent=None, left=None, right=None):
            self._element = element
            self._parent = parent
            self._left = left
            self._right = right
    
    class Position(BinaryTree.Position):
        """An abstraction representing the location of a single element"""
        
        def __init__(self, container, node):
            """Constructor should not be invoked by user"""
            self._container = container
            self._node = node
        
        def element(self):
            """Return the element stored at this Position"""
            return self._node._element
        
        def __eq__(self, other):
            """Return True if other is a Position representing the same location"""
            return type(other) is type(self) and other._node is self._node
    
    def _validate(self, p):
        """Return associated node, if position is valid"""
        if not isinstance(p, self.Position):
            raise TypeError('p must be proper Position type')
        if p._container is not self:
            raise ValueError('p does not belong to this container')
        if p._node._parent is p._node:  # convention for deprecated nodes
            raise ValueError('p is no longer valid')
        return p._node
    
    def _make_position(self, node):
        """Return Position instance for given node (or None if no node)"""
        return self.Position(self, node) if node is not None else None
    
    # Binary tree constructor
    def __init__(self):
        """Create an initially empty binary tree"""
        self._root = None
        self._size = 0
    
    # Public accessors
    def __len__(self):
        """Return the total number of elements in the tree"""
        return self._size
    
    def root(self):
        """Return the root Position of the tree (or None if tree is empty)"""
        return self._make_position(self._root)
    
    def parent(self, p):
        """Return the Position of p's parent (or None if p is root)"""
        node = self._validate(p)
        return self._make_position(node._parent)
    
    def left(self, p):
        """Return the Position of p's left child (or None if no left child)"""
        node = self._validate(p)
        return self._make_position(node._left)
    
    def right(self, p):
        """Return the Position of p's right child (or None if no right child)"""
        node = self._validate(p)
        return self._make_position(node._right)
    
    def num_children(self, p):
        """Return the number of children of Position p"""
        node = self._validate(p)
        count = 0
        if node._left is not None:
            count += 1
        if node._right is not None:
            count += 1
        return count
    
    # Nonpublic update methods
    def _add_root(self, e):
        """Place element e at the root of an empty tree and return new Position"""
        if self._root is not None:
            raise ValueError('Root exists')
        self._size = 1
        self._root = self._Node(e)
        return self._make_position(self._root)
    
    def _add_left(self, p, e):
        """Create a new left child for Position p, storing element e"""
        node = self._validate(p)
        if node._left is not None:
            raise ValueError('Left child exists')
        self._size += 1
        node._left = self._Node(e, node)  # node is its parent
        return self._make_position(node._left)
    
    def _add_right(self, p, e):
        """Create a new right child for Position p, storing element e"""
        node = self._validate(p)
        if node._right is not None:
            raise ValueError('Right child exists')
        self._size += 1
        node._right = self._Node(e, node)  # node is its parent
        return self._make_position(node._right)
    
    def _replace(self, p, e):
        """Replace the element at position p with e, and return old element"""
        node = self._validate(p)
        old = node._element
        node._element = e
        return old
    
    def _delete(self, p):
        """Delete the node at Position p, and replace it with its child, if any"""
        node = self._validate(p)
        if self.num_children(p) == 2:
            raise ValueError('Position has two children')
        child = node._left if node._left else node._right  # might be None
        if child is not None:
            child._parent = node._parent  # child's grandparent becomes parent
        if node is self._root:
            self._root = child  # child becomes root
        else:
            parent = node._parent
            if node is parent._left:
                parent._left = child
            else:
                parent._right = child
        self._size -= 1
        node._parent = node  # convention for deprecated node
        return node._element
    
    def _attach(self, p, t1, t2):
        """Attach trees t1 and t2 as left and right subtrees of external p"""
        node = self._validate(p)
        if not self.is_leaf(p):
            raise ValueError('position must be leaf')
        if not type(self) is type(t1) is type(t2):  # all 3 trees must be same type
            raise TypeError('Tree types must match')
        self._size += len(t1) + len(t2)
        if not t1.is_empty():  # attached t1 as left subtree of node
            t1._root._parent = node
            node._left = t1._root
            t1._root = None  # set t1 instance to empty
            t1._size = 0
        if not t2.is_empty():  # attached t2 as right subtree of node
            t2._root._parent = node
            node._right = t2._root
            t2._root = None  # set t2 instance to empty
            t2._size = 0
'''

# Binary Search Tree implementation
binary_search_tree_py = '''# Binary Search Tree Implementation

class BSTNode:
    """Lightweight class for storing BST nodes"""
    
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class LinkedBST:
    """Linked Binary Search Tree implementation"""
    
    def __init__(self):
        """Initialize an empty binary search tree"""
        self._root = None
        self._size = 0
    
    def __len__(self):
        """Return the number of items in the tree"""
        return self._size
    
    def is_empty(self):
        """Check if the tree is empty"""
        return len(self) == 0
    
    def __contains__(self, item):
        """Return True if item is in the tree"""
        return self._contains(self._root, item)
    
    def _contains(self, node, item):
        """Helper method for __contains__"""
        if node is None:
            return False
        elif item == node.data:
            return True
        elif item < node.data:
            return self._contains(node.left, item)
        else:
            return self._contains(node.right, item)
    
    def find(self, item):
        """Return item if found or None if not found"""
        return self._find(self._root, item)
    
    def _find(self, node, item):
        """Helper method for find"""
        if node is None:
            return None
        elif item == node.data:
            return node.data
        elif item < node.data:
            return self._find(node.left, item)
        else:
            return self._find(node.right, item)
    
    def add(self, item):
        """Add item to the tree"""
        if self.is_empty():
            self._root = BSTNode(item)
        else:
            self._add(self._root, item)
        self._size += 1
    
    def _add(self, node, item):
        """Helper method to add item to BST"""
        if item < node.data:
            if node.left is None:
                node.left = BSTNode(item)
            else:
                self._add(node.left, item)
        else:
            if node.right is None:
                node.right = BSTNode(item)
            else:
                self._add(node.right, item)
    
    def remove(self, item):
        """Remove item from the tree and return it if found, otherwise None"""
        if self.is_empty():
            return None
        else:
            removed_item = self.find(item)
            if removed_item:
                self._root = self._remove(self._root, item)
                self._size -= 1
            return removed_item
    
    def _remove(self, node, item):
        """Helper method to remove item from BST"""
        if node is None:
            return None
        elif item < node.data:
            node.left = self._remove(node.left, item)
            return node
        elif item > node.data:
            node.right = self._remove(node.right, item)
            return node
        else:  # item == node.data
            if node.left is None and node.right is None:  # Leaf node
                return None
            elif node.left is None:  # One right child
                return node.right
            elif node.right is None:  # One left child
                return node.left
            else:  # Two children
                # Find successor (smallest item in right subtree)
                successor = self._find_min(node.right)
                node.data = successor  # Replace with successor
                node.right = self._remove(node.right, successor)  # Remove successor
                return node
    
    def _find_min(self, node):
        """Find minimum value in subtree rooted at node"""
        current = node
        while current.left is not None:
            current = current.left
        return current.data
    
    def inorder(self):
        """Generate an inorder iteration of items in the tree"""
        return self._inorder(self._root)
    
    def _inorder(self, node):
        """Helper method for inorder traversal"""
        items = []
        def recurse(node):
            if node is not None:
                recurse(node.left)
                items.append(node.data)
                recurse(node.right)
        recurse(node)
        return iter(items)
    
    def preorder(self):
        """Generate a preorder iteration of items in the tree"""
        return self._preorder(self._root)
    
    def _preorder(self, node):
        """Helper method for preorder traversal"""
        items = []
        def recurse(node):
            if node is not None:
                items.append(node.data)
                recurse(node.left)
                recurse(node.right)
        recurse(node)
        return iter(items)
    
    def postorder(self):
        """Generate a postorder iteration of items in the tree"""
        return self._postorder(self._root)
    
    def _postorder(self, node):
        """Helper method for postorder traversal"""
        items = []
        def recurse(node):
            if node is not None:
                recurse(node.left)
                recurse(node.right)
                items.append(node.data)
        recurse(node)
        return iter(items)
    
    def __iter__(self):
        """Generate an inorder iteration of items in the tree"""
        return self.inorder()
'''

# Graph implementation
graph_py = '''# Graph Implementation

class Vertex:
    """Lightweight vertex structure for a graph"""
    
    def __init__(self, label=None):
        """Initialize a vertex, optionally with a label"""
        self._label = label
        self._marked = False
    
    def is_marked(self):
        """Return True if the vertex is marked"""
        return self._marked
    
    def set_mark(self):
        """Set the vertex mark to True"""
        self._marked = True
    
    def clear_mark(self):
        """Set the vertex mark to False"""
        self._marked = False
    
    def get_label(self):
        """Return the label associated with this vertex"""
        return self._label
    
    def set_label(self, label):
        """Set the label associated with this vertex"""
        self._label = label
    
    def __eq__(self, other):
        """Two vertices are equal if they have the same labels"""
        if self is other:
            return True
        if type(self) != type(other):
            return False
        return self._label == other._label
    
    def __hash__(self):
        """Vertex hash method needed for dictionary keys"""
        return hash(self._label)
    
    def __str__(self):
        """String representation of the vertex"""
        return str(self._label)


class Edge:
    """Lightweight edge structure for a graph"""
    
    def __init__(self, source, target, weight=None):
        """Initialize an edge from source to target with optional weight"""
        self._source = source
        self._target = target
        self._weight = weight  # weight is optional
        self._marked = False
    
    def is_marked(self):
        """Return True if the edge is marked"""
        return self._marked
    
    def set_mark(self):
        """Set the edge mark to True"""
        self._marked = True
    
    def clear_mark(self):
        """Set the edge mark to False"""
        self._marked = False
    
    def get_source(self):
        """Return the source vertex of this edge"""
        return self._source
    
    def get_target(self):
        """Return the target vertex of this edge"""
        return self._target
    
    def get_weight(self):
        """Return the weight associated with this edge"""
        return self._weight
    
    def set_weight(self, weight):
        """Set the weight associated with this edge"""
        self._weight = weight
    
    def __eq__(self, other):
        """Two edges are equal if they connect the same vertices"""
        if self is other:
            return True
        if type(self) != type(other):
            return False
        return self._source == other._source and \
               self._target == other._target
    
    def __hash__(self):
        """Edge hash method needed for dictionary keys"""
        return hash((self._source, self._target))
    
    def __str__(self):
        """String representation of the edge"""
        result = str(self._source) + ">" + str(self._target)
        if self._weight is not None:
            result += ":" + str(self._weight)
        return result


class LinkedDirectedGraph:
    """A graph implemented with an adjacency list"""
    
    def __init__(self):
        """Create an empty graph"""
        self._vertices = {}  # Maps vertex labels to vertex objects
        self._edges = {}     # Maps vertex labels to dictionary of incident edges
    
    def __str__(self):
        """Returns the string representation of the graph"""
        result = str(len(self._vertices)) + " Vertices: "
        for vertex in self._vertices.values():
            result += str(vertex) + " "
        result += "\\n"
        result += str(self.edge_count()) + " Edges: "
        for edge in self.get_edges():
            result += str(edge) + " "
        return result
    
    def clear_marks(self):
        """Clear all vertex and edge marks"""
        for vertex in self._vertices.values():
            vertex.clear_mark()
        for edges in self._edges.values():
            for edge in edges.values():
                edge.clear_mark()
    
    def clear_vertex_marks(self):
        """Clear all vertex marks"""
        for vertex in self._vertices.values():
            vertex.clear_mark()
    
    def clear_edge_marks(self):
        """Clear all edge marks"""
        for edges in self._edges.values():
            for edge in edges.values():
                edge.clear_mark()
    
    def add_vertex(self, label):
        """Add a vertex with the given label to the graph"""
        if label not in self._vertices:
            self._vertices[label] = Vertex(label)
            self._edges[label] = {}
        return self._vertices[label]
    
    def remove_vertex(self, label):
        """Remove a vertex with the given label from the graph"""
        if label in self._vertices:
            # Remove all edges connected to this vertex
            for target_label in list(self._edges.keys()):
                if label in self._edges[target_label]:
                    del self._edges[target_label][label]
            
            # Remove the vertex and its outgoing edges
            del self._vertices[label]
            del self._edges[label]
    
    def add_edge(self, source_label, target_label, weight=None):
        """Add an edge from the source vertex to the target vertex"""
        # Add the vertices if they don't exist
        if source_label not in self._vertices:
            self.add_vertex(source_label)
        if target_label not in self._vertices:
            self.add_vertex(target_label)
        
        # Add the edge
        self._edges[source_label][target_label] = Edge(
            self._vertices[source_label],
            self._vertices[target_label],
            weight
        )
    
    def remove_edge(self, source_label, target_label):
        """Remove an edge from the source vertex to the target vertex"""
        if source_label in self._edges and target_label in self._edges[source_label]:
            del self._edges[source_label][target_label]
    
    def get_edge(self, source_label, target_label):
        """Return the edge from the source vertex to the target vertex"""
        if source_label in self._edges and target_label in self._edges[source_label]:
            return self._edges[source_label][target_label]
        return None
    
    def get_vertex(self, label):
        """Return the vertex with the given label"""
        if label in self._vertices:
            return self._vertices[label]
        return None
    
    def get_vertices(self):
        """Return an iterator over all vertices in the graph"""
        return iter(self._vertices.values())
    
    def get_edges(self):
        """Return an iterator over all edges in the graph"""
        edges = []
        for source_edges in self._edges.values():
            edges.extend(source_edges.values())
        return iter(edges)
    
    def vertex_count(self):
        """Return the number of vertices in the graph"""
        return len(self._vertices)
    
    def edge_count(self):
        """Return the number of edges in the graph"""
        count = 0
        for source_edges in self._edges.values():
            count += len(source_edges)
        return count
    
    def neighboring_vertices(self, label):
        """Return an iterator over the neighboring vertices of the vertex with the given label"""
        if label in self._edges:
            return iter([self._vertices[target_label] for target_label in self._edges[label]])
        return iter([])
    
    def incident_edges(self, label):
        """Return an iterator over all incident edges of the vertex with the given label"""
        if label in self._edges:
            return iter(self._edges[label].values())
        return iter([])
    
    def is_adjacent(self, source_label, target_label):
        """Return True if the vertices with the given labels are adjacent"""
        return source_label in self._edges and target_label in self._edges[source_label]
'''

# Graph algorithms implementation
graph_algorithms_py = '''# Graph Algorithms

from queue import Queue
from stack import Stack


# Depth-First Search
def dfs(graph, start_vertex, process=print):
    """Perform a depth-first search traversal of the graph starting from start_vertex"""
    # Mark all vertices as unvisited
    graph.clear_vertex_marks()
    
    # Helper function for recursive DFS
    def dfs_helper(vertex):
        # Process the current vertex
        process(vertex)
        vertex.set_mark()  # Mark as visited
        
        # Visit all unvisited neighbors
        for neighbor in graph.neighboring_vertices(vertex.get_label()):
            if not neighbor.is_marked():
                dfs_helper(neighbor)
    
    # Start DFS from the start vertex
    start = graph.get_vertex(start_vertex)
    if start:
        dfs_helper(start)


def dfs_iterative(graph, start_vertex, process=print):
    """Perform a depth-first search traversal using a stack"""
    # Mark all vertices as unvisited
    graph.clear_vertex_marks()
    
    # Create a stack for DFS
    stack = Stack()
    start = graph.get_vertex(start_vertex)
    if not start:
        return
    
    # Push the starting vertex
    stack.push(start)
    
    while not stack.is_empty():
        # Pop a vertex from stack
        vertex = stack.pop()
        
        # If the vertex is not visited yet
        if not vertex.is_marked():
            # Process the vertex
            process(vertex)
            vertex.set_mark()  # Mark as visited
            
            # Push all unvisited neighbors to stack
            for neighbor in graph.neighboring_vertices(vertex.get_label()):
                if not neighbor.is_marked():
                    stack.push(neighbor)


def dfs_complete(graph, process=print):
    """Perform a complete DFS traversal visiting all vertices"""
    # Mark all vertices as unvisited
    graph.clear_vertex_marks()
    
    # Traverse each unvisited vertex
    for vertex in graph.get_vertices():
        if not vertex.is_marked():
            dfs(graph, vertex.get_label(), process)


# Breadth-First Search
def bfs(graph, start_vertex, process=print):
    """Perform a breadth-first search traversal"""
    # Mark all vertices as unvisited
    graph.clear_vertex_marks()
    
    # Create a queue for BFS
    queue = Queue()
    start = graph.get_vertex(start_vertex)
    if not start:
        return
    
    # Mark the source vertex as visited and enqueue it
    start.set_mark()
    queue.put(start)
    
    while not queue.empty():
        # Dequeue a vertex from queue
        vertex = queue.get()
        
        # Process the vertex
        process(vertex)
        
        # Get all adjacent vertices of the dequeued vertex
        # If adjacent is not visited, then mark it visited and enqueue it
        for neighbor in graph.neighboring_vertices(vertex.get_label()):
            if not neighbor.is_marked():
                neighbor.set_mark()
                queue.put(neighbor)


def bfs_complete(graph, process=print):
    """Perform a complete BFS traversal of the graph"""
    # Mark all vertices as unvisited
    graph.clear_vertex_marks()
    
    # Traverse each unvisited vertex
    for vertex in graph.get_vertices():
        if not vertex.is_marked():
            bfs(graph, vertex.get_label(), process)


# Topological Sort
def topological_sort_kahn(graph):
    """Perform a topological sort using Kahn's algorithm"""
    # Create a dictionary to store in-degrees of all vertices
    in_degree = {vertex.get_label(): 0 for vertex in graph.get_vertices()}
    
    # Calculate in-degrees for all vertices
    for edge in graph.get_edges():
        target_label = edge.get_target().get_label()
        in_degree[target_label] += 1
    
    # Create a queue and enqueue all vertices with in-degree 0
    queue = Queue()
    for vertex in graph.get_vertices():
        if in_degree[vertex.get_label()] == 0:
            queue.put(vertex)
    
    # Initialize result list
    topo_order = []
    
    # Process vertices in the queue
    while not queue.empty():
        vertex = queue.get()
        topo_order.append(vertex.get_label())
        
        # Decrease in-degree of adjacent vertices
        for neighbor in graph.neighboring_vertices(vertex.get_label()):
            neighbor_label = neighbor.get_label()
            in_degree[neighbor_label] -= 1
            
            # If in-degree becomes 0, add to queue
            if in_degree[neighbor_label] == 0:
                queue.put(neighbor)
    
    # If not all vertices are included, there's a cycle
    if len(topo_order) != graph.vertex_count():
        return None
    
    return topo_order


def topological_sort_dfs(graph):
    """Perform a topological sort using DFS"""
    # Mark all vertices as unvisited
    graph.clear_vertex_marks()
    
    # Stack to store the topological order
    stack = Stack()
    
    # To track if the graph has a cycle
    visiting = set()
    visited = set()
    
    def dfs_topological_sort(vertex):
        """Helper function for DFS-based topological sort"""
        label = vertex.get_label()
        
        # If vertex is currently being visited, there's a cycle
        if label in visiting:
            return False
        
        # If vertex is already visited, skip
        if label in visited:
            return True
        
        # Mark vertex as being visited
        visiting.add(label)
        
        # Visit all neighbors
        for neighbor in graph.neighboring_vertices(label):
            if not dfs_topological_sort(neighbor):
                return False
        
        # Remove from visiting set and add to visited set
        visiting.remove(label)
        visited.add(label)
        
        # Add to stack
        stack.push(label)
        
        return True
    
    # Visit all vertices
    for vertex in graph.get_vertices():
        if vertex.get_label() not in visited:
            if not dfs_topological_sort(vertex):
                return None  # Graph has a cycle
    
    # Convert stack to list (reverse order)
    topo_order = []
    while not stack.is_empty():
        topo_order.append(stack.pop())
    
    return topo_order


# Shortest Path Algorithms
def dijkstra_shortest_path(graph, start_vertex):
    """Find the shortest path from start_vertex to all other vertices"""
    # Initialize distances with infinity for all vertices except the start vertex
    distances = {vertex.get_label(): float('inf') for vertex in graph.get_vertices()}
    distances[start_vertex] = 0
    
    # Dictionary to store the predecessor of each vertex in the shortest path
    predecessors = {vertex.get_label(): None for vertex in graph.get_vertices()}
    
    # Set of unvisited vertices
    unvisited = set(vertex.get_label() for vertex in graph.get_vertices())
    
    # While there are still unvisited vertices
    while unvisited:
        # Find the unvisited vertex with the smallest distance
        current = min(unvisited, key=lambda v: distances[v])
        
        # If the smallest distance is infinity, the remaining vertices are not reachable
        if distances[current] == float('inf'):
            break
        
        # Remove the current vertex from the unvisited set
        unvisited.remove(current)
        
        # Update distances to all neighboring vertices
        for edge in graph.incident_edges(current):
            neighbor = edge.get_target().get_label()
            weight = edge.get_weight()
            
            if weight is None:  # Skip unweighted edges
                continue
            
            # Calculate new distance
            new_distance = distances[current] + weight
            
            # If the new distance is shorter than the current distance
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current
    
    return distances, predecessors


def bellman_ford_shortest_path(graph, start_vertex):
    """Find the shortest path from start_vertex to all other vertices"""
    # Initialize distances with infinity for all vertices except the start vertex
    distances = {vertex.get_label(): float('inf') for vertex in graph.get_vertices()}
    distances[start_vertex] = 0
    
    # Dictionary to store the predecessor of each vertex in the shortest path
    predecessors = {vertex.get_label(): None for vertex in graph.get_vertices()}
    
    # Relax edges repeatedly
    vertex_count = graph.vertex_count()
    
    # Perform |V|-1 relaxations
    for _ in range(vertex_count - 1):
        # Relax all edges
        for edge in graph.get_edges():
            source = edge.get_source().get_label()
            target = edge.get_target().get_label()
            weight = edge.get_weight()
            
            if weight is None:  # Skip unweighted edges
                continue
            
            # Relaxation step
            if distances[source] != float('inf') and distances[source] + weight < distances[target]:
                distances[target] = distances[source] + weight
                predecessors[target] = source
    
    # Check for negative cycles
    has_negative_cycle = False
    for edge in graph.get_edges():
        source = edge.get_source().get_label()
        target = edge.get_target().get_label()
        weight = edge.get_weight()
        
        if weight is None:  # Skip unweighted edges
            continue
        
        # If we can still relax an edge, there's a negative cycle
        if distances[source] != float('inf') and distances[source] + weight < distances[target]:
            has_negative_cycle = True
            break
    
    return distances, predecessors, has_negative_cycle


def floyd_warshall_all_pairs_shortest_paths(graph):
    """Find the shortest paths between all pairs of vertices"""
    # Get all vertex labels
    vertices = [v.get_label() for v in graph.get_vertices()]
    n = len(vertices)
    
    # Initialize distances and next_vertex matrices
    distances = {}
    next_vertex = {}
    
    # Initialize distances with infinity
    for u in vertices:
        for v in vertices:
            distances[(u, v)] = float('inf')
            next_vertex[(u, v)] = None
    
    # Distance from a vertex to itself is 0
    for v in vertices:
        distances[(v, v)] = 0
    
    # Initialize direct edge weights
    for edge in graph.get_edges():
        u = edge.get_source().get_label()
        v = edge.get_target().get_label()
        weight = edge.get_weight()
        
        if weight is not None:
            distances[(u, v)] = weight
            next_vertex[(u, v)] = v
    
    # Floyd-Warshall algorithm
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if distances[(i, k)] + distances[(k, j)] < distances[(i, j)]:
                    distances[(i, j)] = distances[(i, k)] + distances[(k, j)]
                    next_vertex[(i, j)] = next_vertex[(i, k)]
    
    # Check for negative cycles
    has_negative_cycle = False
    for v in vertices:
        if distances[(v, v)] < 0:
            has_negative_cycle = True
            break
    
    return distances, next_vertex, has_negative_cycle


# Minimum Spanning Tree Algorithms
def prim_minimum_spanning_tree(graph):
    """Find a minimum spanning tree using Prim's algorithm"""
    # Get all vertices
    vertices = list(graph.get_vertices())
    if not vertices:
        return set()
    
    # Start with any vertex
    start = vertices[0].get_label()
    
    # Set of vertices included in MST
    included = {start}
    
    # Set of edges in the MST
    mst_edges = set()
    
    # While not all vertices are included
    while len(included) < len(vertices):
        min_edge = None
        min_weight = float('inf')
        
        # Find the minimum weight edge connecting an included vertex to an unincluded vertex
        for u in included:
            for edge in graph.incident_edges(u):
                v = edge.get_target().get_label()
                weight = edge.get_weight()
                
                if v not in included and weight is not None and weight < min_weight:
                    min_edge = edge
                    min_weight = weight
        
        # If no edge found, graph is not connected
        if min_edge is None:
            break
        
        # Add the target vertex to included set
        included.add(min_edge.get_target().get_label())
        
        # Add the edge to MST
        mst_edges.add(min_edge)
    
    return mst_edges


def kruskal_minimum_spanning_tree(graph):
    """Find a minimum spanning tree using Kruskal's algorithm"""
    # Get all edges and sort them by weight
    edges = [(edge.get_weight(), edge) for edge in graph.get_edges() if edge.get_weight() is not None]
    edges.sort()
    
    # Initialize disjoint set for each vertex
    parent = {}
    rank = {}
    
    def make_set(vertex):
        parent[vertex] = vertex
        rank[vertex] = 0
    
    def find(vertex):
        if parent[vertex] != vertex:
            parent[vertex] = find(parent[vertex])
        return parent[vertex]
    
    def union(vertex1, vertex2):
        root1 = find(vertex1)
        root2 = find(vertex2)
        
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1
    
    # Make a set for each vertex
    for vertex in graph.get_vertices():
        make_set(vertex.get_label())
    
    # Set of edges in the MST
    mst_edges = set()
    
    # Process edges in order of increasing weight
    for weight, edge in edges:
        u = edge.get_source().get_label()
        v = edge.get_target().get_label()
        
        # If including this edge doesn't create a cycle
        if find(u) != find(v):
            mst_edges.add(edge)
            union(u, v)
    
    return mst_edges
'''

# Sorting algorithms implementation
sorting_algorithms_py = '''# Sorting Algorithms Implementation

def bubble_sort(arr):
    """
    Bubble Sort implementation - O(n^2)
    Repeatedly steps through the list, compares adjacent elements and swaps them if they are in wrong order.
    """
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    for i in range(n):
        # Flag to optimize if no swaps occur
        swapped = False
        for j in range(0, n-i-1):
            if arr_copy[j] > arr_copy[j+1]:
                arr_copy[j], arr_copy[j+1] = arr_copy[j+1], arr_copy[j]
                swapped = True
        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break
    return arr_copy


def selection_sort(arr):
    """
    Selection Sort implementation - O(n^2)
    Finds the minimum element from unsorted part and puts it at the beginning.
    """
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr_copy[j] < arr_copy[min_idx]:
                min_idx = j
        # Swap the found minimum element with the first element
        arr_copy[i], arr_copy[min_idx] = arr_copy[min_idx], arr_copy[i]
    return arr_copy


def insertion_sort(arr):
    """
    Insertion Sort implementation - O(n^2)
    Builds the sorted array one item at a time.
    """
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    
    for i in range(1, len(arr_copy)):
        key = arr_copy[i]
        j = i - 1
        # Move elements greater than key to one position ahead
        while j >= 0 and arr_copy[j] > key:
            arr_copy[j + 1] = arr_copy[j]
            j -= 1
        arr_copy[j + 1] = key
    return arr_copy


def merge_sort(arr):
    """
    Merge Sort implementation - O(n log n)
    Divide and conquer algorithm that divides array into two halves, sorts them and merges.
    """
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    
    if len(arr_copy) <= 1:
        return arr_copy

    # Divide array into two halves
    mid = len(arr_copy) // 2
    left = arr_copy[:mid]
    right = arr_copy[mid:]

    # Recursively sort both halves
    left = merge_sort(left)
    right = merge_sort(right)

    # Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    """Helper function for merge sort."""
    result = []
    i = j = 0

    # Merge the two sorted arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):
    """
    Quick Sort implementation - O(n log n) average case, O(n^2) worst case
    Divide and conquer algorithm that picks a pivot and partitions array around it.
    """
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    
    if len(arr_copy) <= 1:
        return arr_copy

    # Helper function for partitioning
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    # Helper function for quick sort recursive implementation
    def quick_sort_helper(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort_helper(arr, low, pi - 1)
            quick_sort_helper(arr, pi + 1, high)

    quick_sort_helper(arr_copy, 0, len(arr_copy) - 1)
    return arr_copy


def heap_sort(arr):
    """
    Heap Sort implementation - O(n log n)
    Sorts by building a max heap and repeatedly extracting the maximum element.
    """
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        # Check if left child exists and is greater than root
        if left < n and arr[largest] < arr[left]:
            largest = left

        # Check if right child exists and is greater than root
        if right < n and arr[largest] < arr[right]:
            largest = right

        # Change root if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr_copy)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr_copy, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr_copy[0], arr_copy[i] = arr_copy[i], arr_copy[0]
        heapify(arr_copy, i, 0)

    return arr_copy


def counting_sort(arr):
    """
    Counting Sort implementation - O(n+k) where k is the range of the input
    Works well for small ranges of positive integers.
    """
    if not arr:
        return []
    
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    
    # Find the maximum element in the input array
    max_element = max(arr_copy)
    min_element = min(arr_copy)
    range_of_elements = max_element - min_element + 1
    
    # Create a count array to store count of individual elements
    count_arr = [0 for _ in range(range_of_elements)]
    output_arr = [0 for _ in range(len(arr_copy))]
    
    # Store count of each element
    for i in range(len(arr_copy)):
        count_arr[arr_copy[i] - min_element] += 1
    
    # Change count_arr[i] so that count_arr[i] now contains actual
    # position of this element in output array
    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i - 1]
    
    # Build the output array
    for i in range(len(arr_copy) - 1, -1, -1):
        output_arr[count_arr[arr_copy[i] - min_element] - 1] = arr_copy[i]
        count_arr[arr_copy[i] - min_element] -= 1
    
    return output_arr


def radix_sort(arr):
    """
    Radix Sort implementation - O(d*(n+k)) where d is number of digits and k is the range
    Works well for integers by sorting them digit by digit.
    """
    if not arr:
        return []
    
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    
    # Find the maximum number to know number of digits
    max_num = max(arr_copy)
    
    # Do counting sort for every digit
    exp = 1
    while max_num // exp > 0:
        # Count sort for current digit
        n = len(arr_copy)
        output = [0] * n
        count = [0] * 10
        
        # Store count of occurrences
        for i in range(n):
            index = arr_copy[i] // exp
            count[index % 10] += 1
            
        # Change count[i] so that count[i] contains actual position of this digit
        for i in range(1, 10):
            count[i] += count[i - 1]
            
        # Build the output array
        for i in range(n - 1, -1, -1):
            index = arr_copy[i] // exp
            output[count[index % 10] - 1] = arr_copy[i]
            count[index % 10] -= 1
            
        # Copy the output array to arr_copy
        for i in range(n):
            arr_copy[i] = output[i]
            
        # Move to next digit
        exp *= 10
        
    return arr_copy


def shell_sort(arr):
    """
    Shell Sort implementation - Between O(n) and O(n^2)
    Modification of insertion sort that allows exchange of far elements.
    """
    # Create a copy to avoid modifying the original
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    # Start with a big gap, then reduce the gap
    gap = n // 2
    
    while gap > 0:
        # Do a gapped insertion sort
        for i in range(gap, n):
            # Save arr_copy[i] in temp and make a hole at position i
            temp = arr_copy[i]
            
            # Shift earlier gap-sorted elements up until the correct location for arr_copy[i] is found
            j = i
            while j >= gap and arr_copy[j - gap] > temp:
                arr_copy[j] = arr_copy[j - gap]
                j -= gap
                
            # Put temp in its correct location
            arr_copy[j] = temp
            
        # Reduce the gap
        gap //= 2
        
    return arr_copy
'''

# Examples implementation
examples_py = '''# Examples of using data structures and algorithms
from stack import Stack, check_balanced_parentheses, evaluate_postfix
from queue import Queue
from binary_search_tree import LinkedBST
from graph import LinkedDirectedGraph
from graph_algorithms import (dfs, bfs, topological_sort_kahn, 
                              dijkstra_shortest_path, kruskal_minimum_spanning_tree)
from sorting_algorithms import (bubble_sort, selection_sort, insertion_sort, 
                              merge_sort, quick_sort, heap_sort, counting_sort)
from point import Point
from church import Church


# Binary Search Tree Example: Find kth smallest element
def find_kth_smallest(bst, k):
    """Find the kth smallest element in a BST."""
    if k <= 0 or k > len(bst):
        return None
    
    # Inorder traversal gives elements in sorted order
    elements = []
    for element in bst.inorder():
        elements.append(element)
        if len(elements) == k:
            return elements[-1]
    
    return None


# Graph Example: Find shortest path
def find_shortest_path(graph, start, end):
    """Find the shortest path between start and end vertices in a graph."""
    # Run Dijkstra's algorithm
    distances, predecessors = dijkstra_shortest_path(graph, start)
    
    # If end is not reachable
    if distances[end] == float('inf'):
        return None
    
    # Reconstruct the path
    path = []
    current = end
    
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    # Reverse the path to get start to end
    path.reverse()
    
    return path


# Demo function to run all examples
def run_examples():
    print("Running examples of data structures and algorithms...")
    
    print("\\n1. Point Class Example")
    point1 = Point(3, 4)
    point2 = Point(1, 1)
    print(f"Point 1: {point1}")
    print(f"Point 2: {point2}")
    point1.rotate(90, point2)
    print(f"Point 1 after rotation around point 2: {point1}")
    
    print("\\n2. Church Class Example")
    church = Church("St. Michael's", 1865, "Lviv", "wood")
    print(church)
    print(f"Age of church: {church.get_age(2023)} years")
    
    print("\\n3. Stack Example: Check balanced parentheses")
    print(f"'([])' is balanced: {check_balanced_parentheses('([])')}")
    print(f"'({[]})' is balanced: {check_balanced_parentheses('({[]})')}")
    print(f"'({[]}' is balanced: {check_balanced_parentheses('({[]}')}")
    
    print("\\n4. Stack Example: Evaluate postfix expression")
    print(f"'5 3 + 2 *' evaluates to: {evaluate_postfix('5 3 + 2 *')}")
    
    print("\\n5. Binary Search Tree Example")
    bst = LinkedBST()
    for item in [50, 30, 70, 20, 40, 60, 80]:
        bst.add(item)
    print(f"BST elements in order: {list(bst.inorder())}")
    print(f"3rd smallest element: {find_kth_smallest(bst, 3)}")
    
    print("\\n6. Graph Example")
    graph = LinkedDirectedGraph()
    edges = [('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1), ('B', 'D', 5), 
             ('C', 'D', 8), ('C', 'E', 10), ('D', 'E', 2), ('D', 'F', 6), 
             ('E', 'F', 3)]
    
    for u, v, w in edges:
        graph.add_edge(u, v, w)
    
    print("Graph created with vertices: A, B, C, D, E, F")
    print(f"Shortest path from A to F: {find_shortest_path(graph, 'A', 'F')}")
    
    print("\\n7. Sorting Algorithms Example")
    test_array = [5, 2, 9, 1, 5, 6]
    print(f"Original array: {test_array}")
    print(f"Bubble Sort:    {bubble_sort(test_array)}")
    print(f"Selection Sort: {selection_sort(test_array)}")
    print(f"Insertion Sort: {insertion_sort(test_array)}")
    print(f"Merge Sort:     {merge_sort(test_array)}")
    print(f"Quick Sort:     {quick_sort(test_array)}")
    print(f"Heap Sort:      {heap_sort(test_array)}")
    print(f"Counting Sort:  {counting_sort(test_array)}")
    
    print("\\nAll examples completed!")


if __name__ == "__main__":
    run_examples()
'''

# Dictionary of all files to create
file_contents = {
    'point.py': point_py,
    'church.py': church_py,
    'node.py': node_py,
    'stack.py': stack_py,
    'queue.py': queue_py,
    'deque.py': deque_py,
    'binary_tree.py': binary_tree_py,
    'binary_search_tree.py': binary_search_tree_py,
    'graph.py': graph_py,
    'graph_algorithms.py': graph_algorithms_py,
    'sorting_algorithms.py': sorting_algorithms_py,
    'examples.py': examples_py,
}

# Write all files
for filename, content in file_contents.items():
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

print("All files have been created successfully in the python_dsa_implementations folder!")
print("To run the examples, execute: python python_dsa_implementations/run_examples.py")