#![allow(clippy::redundant_field_names)]
#![allow(dead_code)]
// This Adelson-Velsky and Landis (AVL) tree implementation comes from Knuth's TAOCP textbook,
// volume 3, "Sorting and Searching". Page numbers refer to the 1973 edition. I have used
// Knuth's variable names where possible and replicated the algorithm steps from the book.
//
// In this implementation:
// * the AVL tree acts as an indexed list of integers, with O(log N) insertion and removal
// * there is a "parent" reference at each node
// * there is also a "direction" at each node, such that node == node.parent.child[node.direction]
// * the rank of a node is the total number of nodes in its subtree (including itself)
// * children are numbered 0 and 1, so that rotation procedures can be generic
// * insert and remove operations are not recursive
// * a "head" node is present after the first value is inserted, so that "empty" is not a special case

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Index;

type InternalIndex = usize;
type ExternalIndex = usize;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
enum Direction {
    Left,
    Right,
}
impl Direction {
    pub fn flip(self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}
type Balance = Ordering;
const HEAD_INDEX: usize = 0;

/// AssociativePositionalList is a positional container in which each value is
/// associated with an index, starting at 0 for the first element. Values can be
/// `insert`ed and `remove`d at any index. The value at any index can be accessed with `get`.
/// But unlike other list containers such as [`Vec`], the association between
/// index and value is reversible, and the index for a value may be determined
/// using `find`.
///
/// AssociativePositionalList requires values to be unique (like a set).
/// Inserting the same value more than once has no effect.
///
/// # Methods
///
/// `insert`, `get` and `remove` use indexes, with 0 being the first item in the list.
///
/// `get` returns the value for a given index.
///
/// `find` returns the index for a given value.
///
/// `len` returns the number of items in the list,
/// `is_empty` returns true if the list is empty, and
/// `clear` removes all items from the list.
///
/// `iter` creates an iterator over the list items.
///
/// # Examples
///
/// ```
/// use associative_positional_list::AssociativePositionalList;
///
/// let mut p: AssociativePositionalList<String> = AssociativePositionalList::new();
/// p.insert(0, "Hello".to_string());
/// p.insert(1, "World".to_string());
/// assert_eq!(p.find(&"World".to_string()), Some(1));
/// assert_eq!(p[0], "Hello");
/// p.remove(0);
/// assert_eq!(p[0], "World");
/// assert_eq!(p.find(&"World".to_string()), Some(0));
/// p.remove(0);
/// assert!(p.is_empty());
///
///
/// ```
///
/// # Limitations
///
/// * At least two copies of each value will exist within the container.
/// * Values must be hashable.
/// * Values do not have to be comparable.
///
/// # Time complexity
///
/// The `insert`, `get`, `remove` and `find` operations have logarithmic
/// time complexity (i.e. O(log N) operations are required).
///
/// `len`, `is_empty` and `clear` have constant time.
///
/// # Notes
///
/// This crate was developed by a relative newcomer to Rust as part of a learning exercise.
/// It may not be very efficient. Some of the interfaces you may expect as part of a list
/// container (or a set) are not present.
///
/// # Implementation
///
/// AssociativePositionalList is implemented using a self-balancing binary tree. These are most commonly used
/// to implement ordered associative data structures, similar to [`HashMap`] but with values
/// stored in key order. But they can also be used to implement indexed data structures such
/// as lists, by using the index (or "rank") of each value as the ordering criteria. This
/// is not possible with most generic tree structures (e.g. [`std::collections::BTreeMap`])
/// because they do not provide structural information to the comparison function. Therefore,
/// AssociativePositionalList uses its own binary tree implementation, which is an [AVL] tree based on pseudocode
/// from [Knuth's TAOCP] volume 3, "Sorting and Searching".
///
/// The `find` method uses a [`HashMap`] to determine the tree node corresponding to a value,
/// and then the index of the tree node is computed based on the "rank".
///
/// Insert and remove operations are iterative (no recursion).
///
/// [AVL]: https://en.wikipedia.org/wiki/AVL_tree
/// [Knuth's TAOCP]: https://en.wikipedia.org/wiki/The_Art_of_Computer_Programming
///

#[derive(Default)]
pub struct AssociativePositionalList<ValueType>
where
    ValueType: std::hash::Hash + Eq + Clone,
{
    lookup: HashMap<ValueType, InternalIndex>,
    data: Vec<AVLNode<ValueType>>,
}

#[derive(Debug)]
struct AVLNode<ValueType> {
    child: [Option<InternalIndex>; 2],
    value: ValueType,
    balance: Balance,
    direction: Direction,
    rank: ExternalIndex,
    parent: InternalIndex,
}

impl<ValueType> AVLNode<ValueType> {
    fn get_child(&self, d: Direction) -> Option<InternalIndex> {
        match d {
            Direction::Left => self.child[0],
            Direction::Right => self.child[1],
        }
    }
    fn get_child_mut(&mut self, d: Direction) -> &mut Option<InternalIndex> {
        match d {
            Direction::Left => &mut self.child[0],
            Direction::Right => &mut self.child[1],
        }
    }
}

impl<ValueType> Index<usize> for AssociativePositionalList<ValueType>
where
    ValueType: std::hash::Hash + Eq + Clone,
{
    /// Get the value at the specified index in the AssociativePositionalList.
    /// Will panic if the index is not less than the length.
    type Output = ValueType;

    fn index(&self, index: usize) -> &Self::Output {
        return self.get(index).unwrap();
    }
}

impl<ValueType> PartialEq for AssociativePositionalList<ValueType>
where
    ValueType: std::hash::Hash + Eq + Clone,
{
    /// Compare the value of a AssociativePositionalList to another.
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let mut it1 = self.iter();
        let mut it2 = other.iter();
        loop {
            match (it1.next(), it2.next()) {
                (None, None) => {
                    return true; // reached the end
                }
                (Some(x), Some(y)) => {
                    if x != y {
                        return false; // found elements that don't match
                    }
                }
                _ => {
                    return false; // reached the end with one, but not the other
                }
            }
        }
    }
}

impl<ValueType> Debug for AssociativePositionalList<ValueType>
where
    ValueType: std::hash::Hash + Eq + Clone + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        return f.debug_list().entries(self.iter()).finish();
    }
}

impl<ValueType> FromIterator<ValueType> for AssociativePositionalList<ValueType>
where
    ValueType: std::hash::Hash + Eq + Clone,
{
    fn from_iter<I: IntoIterator<Item = ValueType>>(
        iter: I,
    ) -> AssociativePositionalList<ValueType> {
        let mut p: AssociativePositionalList<ValueType> = AssociativePositionalList::new();
        for (i, x) in iter.into_iter().enumerate() {
            p.insert(i, x.clone());
        }
        p
    }
}

struct IterStackItem {
    index: InternalIndex,
    direction: Direction,
}

/// This is an iterator over elements in an AssociativePositionalList
pub struct Iter<'a, ValueType: 'a>
where
    ValueType: std::hash::Hash + Eq + Clone,
{
    stack: Vec<IterStackItem>,
    parent: &'a AssociativePositionalList<ValueType>,
}

impl<'a, ValueType> Iterator for Iter<'a, ValueType>
where
    ValueType: std::hash::Hash + Eq + Clone,
{
    type Item = ValueType;

    fn next(&mut self) -> Option<Self::Item> {
        // If the stack is empty, no more items
        if self.stack.is_empty() {
            return None;
        }

        // Find the next item to be returned - the top of the stack is
        // either the last node to be returned by the iterator,
        // or the head of the list
        let c = self.stack.last().unwrap().index;
        let c = self.parent.iget(c).child[1];
        if let Some(c) = c {
            // There is a right child, so we should move right
            self.stack.push(IterStackItem {
                index: c,
                direction: Direction::Right,
            });

            // Fill the stack with the path to the leftmost item with a value
            let mut child = c;
            while let Some(index) = self.parent.iget(child).get_child(Direction::Left) {
                child = index;
                self.stack.push(IterStackItem {
                    index: child,
                    direction: Direction::Left,
                });
            }
        } else {
            // There is no right child, so we should move up
            loop {
                let direction = self.stack.last().unwrap().direction;
                self.stack.pop();
                if direction == Direction::Left {
                    // If we returned from the left, we can move right next time
                    break;
                }
                if self.stack.is_empty() {
                    // If the stack is now empty, this was the last item
                    return None;
                }
            }
        }

        // Return the value referenced at the top of the stack
        let n: &AVLNode<ValueType> = self.parent.iget(self.stack.last().unwrap().index);
        Some(n.value.clone())
    }
}

impl<ValueType> AssociativePositionalList<ValueType>
where
    ValueType: std::hash::Hash + Eq + Clone,
{
    /// Makes a new, empty AssociativePositionalList.
    pub fn new() -> Self {
        AssociativePositionalList {
            data: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    #[track_caller]
    fn iget(&self, index: InternalIndex) -> &AVLNode<ValueType> {
        return self.data.get(index).unwrap();
    }

    fn iget_mut(&mut self, index: InternalIndex) -> &mut AVLNode<ValueType> {
        return self.data.get_mut(index).unwrap();
    }

    fn head(&self) -> &AVLNode<ValueType> {
        if self.data.is_empty() {
            panic!("cannot access head() until one element has been inserted");
        }
        return self.data.get(0).unwrap();
    }
    fn head_mut(&mut self) -> &mut AVLNode<ValueType> {
        if self.data.is_empty() {
            panic!("cannot access head() until one element has been inserted");
        }
        return self.data.get_mut(0).unwrap();
    }

    fn left_rank(&self, index: InternalIndex) -> ExternalIndex {
        match self.iget(index).get_child(Direction::Left) {
            Some(c) => self.iget(c).rank,
            None => 0,
        }
    }

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> bool {
        self.lookup.is_empty()
    }

    /// Returns the number of items in the list
    pub fn len(&self) -> ExternalIndex {
        self.lookup.len()
    }

    /// Returns the index where `value` can be found, or `None` if `value` is not present.
    ///
    /// Note: If values have not always been unique within the list, then the `find` method's
    /// return is not defined.
    pub fn find(&self, value: &ValueType) -> Option<ExternalIndex> {
        let mut p: InternalIndex = *self.lookup.get(value)?;

        if self.iget(p).value != *value {
            //TODO: Panic here?
            return None; // The value has changed, the rule about uniqueness wasn't followed
        }

        let mut ext_index: ExternalIndex = self.left_rank(p);
        let end: InternalIndex = self.head().child[1].unwrap();
        while p != end {
            if self.iget(p).direction == Direction::Right {
                p = self.iget(p).parent;
                ext_index += self.left_rank(p) + 1;
            } else {
                p = self.iget(p).parent;
            }
        }
        Some(ext_index)
    }

    /// Returns a reference to the value at `index`, if `index` is less than the length of the list.
    /// Otherwise returns `None`.
    pub fn get(&self, index: ExternalIndex) -> Option<&ValueType> {
        if self.data.is_empty() {
            // nothing was ever inserted into the list
            return None;
        }
        let mut p = self.head().get_child(Direction::Right);
        let mut ext_index_copy = index;

        loop {
            match p {
                None => return None,
                Some(child) => {
                    let left_rank = self.left_rank(child);
                    match ext_index_copy.cmp(&left_rank) {
                        Ordering::Less => p = self.iget(child).get_child(Direction::Left),
                        Ordering::Equal => return Some(&self.iget(child).value), // index found
                        Ordering::Greater => {
                            ext_index_copy -= left_rank + 1;
                            p = self.iget(child).get_child(Direction::Right);
                        }
                    }
                }
            }
        }
    }

    /// Returns an iterator over all values in list order.
    pub fn iter(&self) -> Iter<ValueType> {
        let mut stack: Vec<IterStackItem> = Vec::new();
        if !self.is_empty() {
            // If the list is non-empty, begin iteration at the head
            stack.push(IterStackItem {
                index: HEAD_INDEX,
                direction: Direction::Right,
            });
        }
        Iter {
            parent: self,
            stack: stack,
        }
    }

    /// Remove all items from the list
    pub fn clear(&mut self) {
        if !self.data.is_empty() {
            // Quickly reset the head of the list
            self.lookup.clear();
            self.data.truncate(HEAD_INDEX + 1);
            self.head_mut().child = [None, None];
        }
    }

    fn new_node(&mut self, value: ValueType, parent: InternalIndex) -> InternalIndex {
        let n: AVLNode<ValueType> = AVLNode {
            child: [None, None],
            value: value,
            balance: Ordering::Equal,
            direction: Direction::Left,
            rank: 0,
            parent: parent,
        };
        self.data.push(n);
        self.data.len() - 1
    }

    fn free_node(&mut self, remove_index: InternalIndex) {
        // Swap with the item at the end
        let replacement: AVLNode<ValueType> = self.data.pop().unwrap();
        let replacement_index: InternalIndex = self.data.len();

        if remove_index >= replacement_index {
            // remove_index was at the end, so nothing more is needed - it's gone!
            return;
        }

        // Change the index of "replacement" to be "remove_index" by making
        // new child-parent links
        if let Some(parent) = self.data.get_mut(replacement.parent) {
            for c in &mut parent.child {
                if *c == Some(replacement_index) {
                    *c = Some(remove_index);
                }
            }
        }
        for c in replacement.child.iter().flatten() {
            if let Some(child) = self.data.get_mut(*c) {
                child.parent = remove_index;
            }
        }

        // Change the index for this value
        self.lookup.insert(replacement.value.clone(), remove_index);

        // replace the node itself
        *self.data.get_mut(remove_index).unwrap() = replacement;
    }

    /// Insert `value` at `index`, causing the indexes of all items with index >= `index`
    /// to be increased by 1.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// * If the set did not previously contain this value, true is returned.
    /// * If the set already contained this value, false is returned.
    pub fn insert(&mut self, index: ExternalIndex, value: ValueType) -> bool {
        if index > self.len() {
            panic!(
                "Can't insert item at index {index}: current length is {}",
                self.len()
            );
        }
        if self.data.is_empty() {
            // Tree has never been used before - add the HEAD_INDEX node
            if self.new_node(value.clone(), HEAD_INDEX) != HEAD_INDEX {
                panic!("index of head node is not HEAD_INDEX");
            }
        }

        let mut p: Option<InternalIndex> = self.head().get_child(Direction::Right); // the pointer variable p will move down the tree
        let mut s: Option<InternalIndex> = self.head().get_child(Direction::Right); // s will point to the place where rebalancing may be necessary
        let mut t: InternalIndex = HEAD_INDEX; // t will always point to the parent of s
        let mut q: Option<InternalIndex>;
        let r: InternalIndex;
        let mut direction: Direction;
        let mut s_index: ExternalIndex = index; // index at the point where rebalancing was necessary
        let mut c_index: ExternalIndex = index;

        match p {
            None => {
                // empty tree special case
                let i = self.new_node(value.clone(), HEAD_INDEX);
                *self.iget_mut(HEAD_INDEX).get_child_mut(Direction::Right) = Some(i);
                let mut n = self.iget_mut(i);
                n.direction = Direction::Right;
                n.rank = 1;
                self.lookup.insert(value, i);
                true
            }
            Some(mut p_inner) => {
                if self.lookup.contains_key(&value) {
                    // value is already present - nothing happens
                    return false;
                }

                loop {
                    if c_index <= self.left_rank(p_inner) {
                        // move left
                        direction = Direction::Left;
                    } else {
                        // move right
                        direction = Direction::Right;
                        c_index -= self.left_rank(p_inner) + 1;
                    }

                    // inserting something below p - therefore, rank of p increases
                    self.iget_mut(p_inner).rank += 1;

                    q = self.iget(p_inner).get_child(direction);
                    match q {
                        Some(q_inner) => {
                            // Continue search
                            if self.iget(q_inner).balance != Ordering::Equal {
                                t = p_inner;
                                s = Some(q_inner);
                                s_index = c_index;
                            }
                            p_inner = q_inner;
                        }
                        None => {
                            // New child (appending)
                            let q_inner = self.new_node(value.clone(), p_inner);
                            q = Some(q_inner);
                            let mut n = self.iget_mut(q_inner);
                            n.direction = direction;
                            n.rank = 1;
                            n.balance = Ordering::Equal;
                            *self.iget_mut(p_inner).get_child_mut(direction) = q;
                            self.lookup.insert(value, q_inner);
                            break;
                        }
                    }
                }

                // adjust balance factors
                assert!(s.is_some());
                let s = s.unwrap();
                c_index = s_index;
                if c_index <= self.left_rank(s) {
                    p = self.iget(s).get_child(Direction::Left);
                    r = p.unwrap();
                } else {
                    c_index -= self.left_rank(s) + 1;
                    p = self.iget(s).get_child(Direction::Right);
                    r = p.unwrap();
                }
                while p != q {
                    if c_index <= self.left_rank(p.unwrap()) {
                        self.iget_mut(p.unwrap()).balance = Ordering::Greater;
                        p = self.iget(p.unwrap()).child[0];
                    } else {
                        c_index -= self.left_rank(p.unwrap()) + 1;
                        self.iget_mut(p.unwrap()).balance = Ordering::Less;
                        p = self.iget(p.unwrap()).child[1];
                    }
                }
                // A7 balancing act
                let a: Ordering;
                if s_index <= self.left_rank(s) {
                    a = Ordering::Greater;
                    direction = Direction::Left;
                } else {
                    a = Ordering::Less;
                    direction = Direction::Right;
                }
                if self.iget(s).balance == Ordering::Equal {
                    // case i. The tree has grown higher
                    self.iget_mut(s).balance = a;
                    return true;
                } else if self.iget(s).balance == a.reverse() {
                    // case ii. The tree has gotten more balanced
                    self.iget_mut(s).balance = Ordering::Equal;
                    return true;
                }
                // case iii. The tree is not balanced
                // note: r = s.get_child(direction)
                if self.iget(r).balance == a {
                    // page 454 case 1
                    p = Some(self.single_rotation(r, s, direction));
                    self.rerank(s);
                    self.rerank(r);
                    self.rerank(p.unwrap());
                } else if self.iget(r).balance == a.reverse() {
                    // page 454 case 2
                    p = Some(self.double_rotation(r, s, direction));
                    self.rerank(s);
                    self.rerank(r);
                    self.rerank(p.unwrap());
                } else {
                    // unbalanced in an unexpected way
                    panic!();
                }
                // A10 finishing touch
                if Some(s) == self.iget(t).get_child(Direction::Right) {
                    self.iget_mut(t).child[1] = p;
                    self.iget_mut(p.unwrap()).parent = t;
                    self.iget_mut(p.unwrap()).direction = Direction::Right;
                } else {
                    self.iget_mut(t).child[0] = p;
                    self.iget_mut(p.unwrap()).parent = t;
                    self.iget_mut(p.unwrap()).direction = Direction::Left;
                }
                true
            }
        }
    }

    fn single_rotation(
        &mut self,
        r: InternalIndex,
        s: InternalIndex,
        direction: Direction,
    ) -> InternalIndex {
        // page 457 A8 single rotation
        // as applied to case 1 (top of page 454) in which s is A and r is B
        // Initially r is a child of s. In the book, direction = 1, as follows:
        //
        //      |               ->            |
        //      s               ->            r
        //    /   \        SingleRotation   /   \
        // alpha   r            ->        s     gamma
        //       /   \          ->      /   \
        //    beta   gamma      ->  alpha   beta
        //
        // direction = 0 is the same operation applied to a mirror image.

        let p = r;
        *self.iget_mut(s).get_child_mut(direction) = self.iget(r).get_child(direction.flip()); // beta subtree moved from r to s
        *self.iget_mut(r).get_child_mut(direction.flip()) = Some(s); // node r becomes child of s
        self.iget_mut(s).balance = Ordering::Equal;
        self.iget_mut(r).balance = Ordering::Equal;
        self.iget_mut(s).direction = direction.flip();
        self.iget_mut(s).parent = r;

        if let Some(c) = self.iget(s).get_child(direction) {
            self.iget_mut(c).parent = s;
            self.iget_mut(c).direction = direction;
        }
        p
    }

    fn double_rotation(
        &mut self,
        r: InternalIndex,
        s: InternalIndex,
        direction: Direction,
    ) -> InternalIndex {
        // A9 double rotation
        // as applied to case 2 (top of page 454) in which s is A, r is B, and p is X
        // Initially r is a child of s. In the book, direction = 1, as follows:
        //
        //         |            ->                     |
        //         s            ->                     p
        //       /   \      DoubleRotation           /    \
        //    alpha   r         ->                 s        r
        //          /   \       ->               /   \    /   \
        //         p    delta   ->           alpha beta gamma delta
        //       /   \          ->
        //     beta  gamma      ->
        //
        // direction = 0 is the same operation applied to a mirror image.

        let a = match direction {
            Direction::Left => Ordering::Greater,
            Direction::Right => Ordering::Less,
        };

        let p: InternalIndex = self.iget(r).get_child(direction.flip()).unwrap(); // p is child of r (node X in the book)
        *self.iget_mut(r).get_child_mut(direction.flip()) = self.iget(p).get_child(direction); // gamma subtree moved from p to r
        *self.iget_mut(p).get_child_mut(direction) = Some(r); // r becomes child of p
        *self.iget_mut(s).get_child_mut(direction) = self.iget(p).get_child(direction.flip()); // beta subtree moved from p to s
        *self.iget_mut(p).get_child_mut(direction.flip()) = Some(s); // s becomes child of p
        if self.iget(p).balance == a {
            self.iget_mut(s).balance = a.reverse();
            self.iget_mut(r).balance = Ordering::Equal;
        } else if self.iget(p).balance == Ordering::Equal {
            self.iget_mut(s).balance = Ordering::Equal;
            self.iget_mut(r).balance = Ordering::Equal;
        } else {
            self.iget_mut(s).balance = Ordering::Equal;
            self.iget_mut(r).balance = a;
        }
        self.iget_mut(p).balance = Ordering::Equal;

        self.iget_mut(s).parent = p;
        self.iget_mut(s).direction = direction.flip();
        if let Some(sc) = self.iget(s).get_child(direction) {
            self.iget_mut(sc).parent = s;
            self.iget_mut(sc).direction = direction;
        }

        self.iget_mut(r).parent = p;
        self.iget_mut(r).direction = direction;
        if let Some(rc) = self.iget(r).get_child(direction.flip()) {
            self.iget_mut(rc).parent = r;
            self.iget_mut(rc).direction = direction.flip();
        }
        p
    }

    fn rerank(&mut self, node: InternalIndex) {
        let child_rank_sum = self
            .iget(node)
            .child
            .iter()
            .flatten()
            .map(|&c| self.iget(c).rank)
            .sum::<usize>();
        self.iget_mut(node).rank = 1 + child_rank_sum;
    }

    /// Removes the value at `index`, causing the indexes of all items with index > `index`
    /// to be decreased by 1. No effect if `index` is not valid.
    pub fn remove(&mut self, index: ExternalIndex) {
        if self.data.is_empty() {
            // nothing was ever inserted into the list
            return;
        }

        let p: Option<InternalIndex> = self.head().get_child(Direction::Right);
        let mut adjust_p: InternalIndex = HEAD_INDEX;
        let mut adjust_direction: Direction = Direction::Right;
        let mut c_index: ExternalIndex = index;

        if p.is_none() || (index >= self.iget(p.unwrap()).rank) {
            // unable to delete element outside of list
            return;
        }
        let mut p = p.unwrap();

        loop {
            // element will be removed below p
            self.iget_mut(p).rank -= 1;
            match c_index.cmp(&self.left_rank(p)) {
                Ordering::Less => {
                    adjust_p = p;
                    adjust_direction = Direction::Left;
                    p = self.iget(p).get_child(Direction::Left).unwrap();
                }
                Ordering::Equal => {
                    // element found - stop
                    break;
                }
                Ordering::Greater => {
                    adjust_p = p;
                    adjust_direction = Direction::Right;
                    c_index -= self.left_rank(p) + 1;
                    p = self.iget(p).get_child(Direction::Right).unwrap();
                }
            }
        }
        let free_before_returning: InternalIndex;

        // found the node to delete (p)
        if self.iget(p).child.iter().all(|c| c.is_some()) {
            // non-leaf node with two children being deleted
            // page 429 Tree deletion (is for a non-balanced binary tree)

            // In this case we find another node with 0 or 1 child which can be
            // deleted instead. We swap this node into the tree.

            // q - the node we would like to remove
            let q = p;
            adjust_p = p;
            adjust_direction = Direction::Right;

            // find p, a node we can actually remove
            p = self.iget(p).get_child(Direction::Right).unwrap();
            while let Some(child) = self.iget(p).get_child(Direction::Left) {
                self.iget_mut(p).rank -= 1;
                adjust_p = p;
                adjust_direction = Direction::Left;
                p = child;
            }
            self.iget_mut(p).rank -= 1;

            // Now we found p, a node with zero or one child - easily removed:
            let p_child_1 = self.iget(p).get_child(Direction::Right);

            // move p's contents to q
            self.lookup.remove(&self.iget(q).value.clone());
            self.lookup.insert(self.iget(p).value.clone(), q);
            self.iget_mut(q).value = self.iget(p).value.clone();
            free_before_returning = p;
            p = q;

            // fix up a connection to p's child (if p had a child)
            *self.iget_mut(adjust_p).get_child_mut(adjust_direction) = p_child_1;
            if let Some(pch1) = p_child_1 {
                self.iget_mut(pch1).parent = adjust_p;
                self.iget_mut(pch1).direction = adjust_direction;
            }
            if let Some(p_left_child) = self.iget(p).get_child(Direction::Left) {
                self.iget_mut(p_left_child).parent = p;
                self.iget_mut(p_left_child).direction = Direction::Left;
            }
            if let Some(p_right_child) = self.iget(p).get_child(Direction::Right) {
                self.iget_mut(p_right_child).parent = p;
                self.iget_mut(p_right_child).direction = Direction::Right;
            }
        } else if let Some(p_left_child) = self.iget(p).get_child(Direction::Left) {
            // Node has one left child - so it's easily removed:
            self.lookup.remove(&self.iget(p).value.clone());
            self.iget_mut(adjust_p).child[adjust_direction as usize] = self.iget(p).child[0];
            self.iget_mut(p_left_child).parent = adjust_p;
            self.iget_mut(p_left_child).direction = adjust_direction;
            free_before_returning = p;
        } else {
            //Node has no children, or a right child - again easily removed.
            self.lookup.remove(&self.iget(p).value.clone());
            *self.iget_mut(adjust_p).get_child_mut(adjust_direction) =
                self.iget(p).get_child(Direction::Right);
            if let Some(p_right_child) = self.iget(p).get_child(Direction::Right) {
                self.iget_mut(p_right_child).parent = adjust_p;
                self.iget_mut(p_right_child).direction = adjust_direction;
            }
            free_before_returning = p;
        }

        // The process of deleting node p sets parent.p.child[parent.direction]
        // and so the balance factor at parent.p is adjusted
        while adjust_p != HEAD_INDEX {
            let next_adjust_direction: Direction = self.iget(adjust_p).direction;
            let next_adjust_p: InternalIndex = self.iget(adjust_p).parent;
            let adjust_a: Ordering = if adjust_direction == Direction::Right {
                Ordering::Less
            } else {
                Ordering::Greater
            };

            if self.iget(adjust_p).balance == adjust_a {
                // page 466 i: repeat adjustment procedure for parent
                self.iget_mut(adjust_p).balance = Ordering::Equal;
            } else if self.iget(adjust_p).balance == Ordering::Equal {
                // page 466 ii: tree is balanced
                self.iget_mut(adjust_p).balance = adjust_a.reverse();
                break;
            } else {
                // page 466 iii - rebalancing required
                let s = adjust_p; // parent of subtree requiring rotation
                let r = self
                    .iget(adjust_p)
                    .get_child(adjust_direction.flip())
                    .unwrap(); // child requiring rotation is the OPPOSITE of the one removed

                if self.iget(r).balance == adjust_a.reverse() {
                    // page 454 case 1
                    p = self.single_rotation(r, s, adjust_direction.flip());
                    *self
                        .iget_mut(next_adjust_p)
                        .get_child_mut(next_adjust_direction) = Some(p);
                    self.iget_mut(p).parent = next_adjust_p;
                    self.iget_mut(p).direction = next_adjust_direction;
                    self.rerank(s);
                    self.rerank(r);
                    self.rerank(p);
                } else if self.iget(r).balance == adjust_a {
                    // page 454 case 2
                    p = self.double_rotation(r, s, adjust_direction.flip());
                    *self
                        .iget_mut(next_adjust_p)
                        .get_child_mut(next_adjust_direction) = Some(p);
                    self.iget_mut(p).parent = next_adjust_p;
                    self.iget_mut(p).direction = next_adjust_direction;
                    self.rerank(s);
                    self.rerank(r);
                    self.rerank(p);
                } else if self.iget(r).balance == Ordering::Equal {
                    // case 3: like case 1 except that beta has height h + 1 (same as gamma)
                    p = self.single_rotation(r, s, adjust_direction.flip());
                    *self
                        .iget_mut(next_adjust_p)
                        .get_child_mut(next_adjust_direction) = Some(p);
                    self.iget_mut(adjust_p).balance = adjust_a.reverse();
                    self.iget_mut(p).balance = adjust_a;
                    self.iget_mut(p).parent = next_adjust_p;
                    self.iget_mut(p).direction = next_adjust_direction;
                    self.rerank(s);
                    self.rerank(r);
                    self.rerank(p);
                    break; // balanced after single rotation
                } else {
                    // unexpected balance value
                    panic!();
                }
            }
            adjust_direction = next_adjust_direction;
            adjust_p = next_adjust_p;
        }
        // Don't free any nodes while we have copies of the indexes, because
        // indexes will be invalidated.
        self.free_node(free_before_returning);
    }
}

impl<V> AssociativePositionalList<V>
where
    V: std::hash::Hash + Eq + Clone + Debug,
{
    //consistency checks.
    fn calculate_depth_at_node(&self, node: InternalIndex) -> usize {
        self.iget(node)
            .child
            .iter()
            .flatten()
            .map(|&c| self.calculate_depth_at_node(c))
            .max()
            .unwrap_or_default()
            + 1
    }
    fn calculate_rank_at_node(&self, node: InternalIndex) -> usize {
        self.iget(node)
            .child
            .iter()
            .copied()
            .flatten()
            .map(|c| self.calculate_rank_at_node(c))
            .sum::<usize>()
            + 1
    }
    fn calculate_balance(&self, node: InternalIndex) -> Ordering {
        let n = self.iget(node);
        let [d1, d2] = [Direction::Left, Direction::Right].map(|d| {
            n.get_child(d)
                .map(|c| self.calculate_depth_at_node(c))
                .unwrap_or_default()
        });
        assert!(d1.abs_diff(d2) <= 1);
        d1.cmp(&d2)
    }
    // Check that the whole tree is internally consistent
    fn check_consistent(&self) {
        //global constraints - emptiness of data and lookup match.
        // dbg!(&self.data);
        if self.data.is_empty() {
            // Tree has never been used - check state
            assert!(self.lookup.is_empty());
            return;
        }
        if let Some(ch) = self.head().get_child(Direction::Right) {
            assert_eq!(self.iget(ch).parent, HEAD_INDEX);
            assert_eq!(self.iget(ch).direction, Direction::Right);
            let mut visited: HashMap<InternalIndex, bool> = HashMap::new();
            self.check_consistent_node(ch, &mut visited);
            assert_eq!(visited.len(), self.len());
        }
    }

    // Check that a subtree (with root 'node') is internally consistent
    // (parent/child links are correct, nodes appear exactly once, balanced,
    // balance and rank values are correct)
    fn check_consistent_node(
        &self,
        node: InternalIndex,
        visited: &mut HashMap<InternalIndex, bool>,
    ) {
        assert!(!visited.contains_key(&node));
        visited.insert(node, true);

        assert!(node < self.data.len());
        let me = self.iget(node);
        for d in [Direction::Left, Direction::Right] {
            if let Some(child) = me.get_child(d) {
                self.check_consistent_node(child, visited);
                assert_eq!(self.iget(child).parent, node);
                assert_eq!(self.iget(child).direction, d);
            }
        }
        assert_eq!(
            me.rank,
            self.calculate_rank_at_node(node),
            "Rank checking node with value {:?} in tree: {}",
            me.value,
            self.draw_tree()
        );
        assert_eq!(
            me.balance,
            self.calculate_balance(node),
            "Balance checking node with value {:?} in tree: {}",
            me.value,
            self.draw_tree()
        );
    }

    pub fn draw_tree(&self) -> String {
        if self.is_empty() {
            return "[]".to_owned();
        }
        let mut ans = String::new();
        let mut stack = vec![(0, self.head().get_child(Direction::Right).unwrap())];
        while let Some((d, n)) = stack.pop() {
            let me = self.iget(n);
            match d {
                0 => {
                    //we're going left.
                    ans.push('[');
                    stack.push((1, n));
                    if let Some(left) = me.get_child(Direction::Left) {
                        stack.push((0, left));
                    }
                }
                1 => {
                    let bal_char = match me.balance {
                        Ordering::Less => '<',
                        Ordering::Equal => '=',
                        Ordering::Greater => '>',
                    };
                    ans.push_str(&format!(",{:?}{},", me.value, bal_char));
                    stack.push((2, n));
                    if let Some(right) = me.get_child(Direction::Right) {
                        stack.push((0, right));
                    }
                }
                2 => {
                    ans.push(']');
                }
                _ => unreachable!(),
            }
        }
        ans
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_equality() {
        let a: AssociativePositionalList<i8> = [1].into_iter().collect();
        let b: AssociativePositionalList<i8> = [].into_iter().collect();
        let c: AssociativePositionalList<i8> = [1].into_iter().collect();
        assert_ne!(a, b);
        assert_ne!(b, a);
        assert_eq!(a, c);
        assert_eq!(c, a);
    }
    #[test]
    fn test_insert() {
        let mut t = AssociativePositionalList::<u16>::new();
        t.insert(0, 1);
        t.insert(1, 2);
        let result: Vec<u16> = t.iter().collect();
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_remove() {
        let mut t = AssociativePositionalList::<u16>::new();
        t.insert(0, 1);
        t.insert(1, 2);
        t.remove(0);
        let result: Vec<u16> = t.iter().collect();
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_rev_list_of_3() {
        let vec = vec![0, 1, 2];
        let mut t = AssociativePositionalList::new();
        for a in vec.iter().rev() {
            t.insert(0, a);
            t.check_consistent();
        }
        let back_to_vec: Vec<u8> = t.iter().copied().collect();
        assert_eq!(back_to_vec, vec);
    }
    #[test]
    fn test_list_of_6() {
        let vec = vec![0, 1, 2, 3, 4, 5];
        let mut t = AssociativePositionalList::new();
        for (ix, a) in vec.iter().enumerate() {
            t.insert(ix, a);
            t.check_consistent();
        }
        let back_to_vec: Vec<u8> = t.iter().copied().collect();
        assert_eq!(back_to_vec, vec);
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn test_add_in_order(size in 0..100u8) {
            let vec : Vec<u8> = (0..size).collect();
            let mut t = AssociativePositionalList::new();
            for (ix, a) in vec.iter().enumerate() {
                t.insert(ix, a);
            }
            let back_to_vec : Vec<u8> = t.iter().copied().collect();
            assert_eq!(back_to_vec, vec);
        }
        #[test]
        fn test_add_in_rev_order(size in 0..100u8) {
            let vec : Vec<u8> = (0..size).collect();
            let mut t = AssociativePositionalList::new();
            for a in vec.iter().rev() {
                t.insert(0, a);
            }
            let back_to_vec : Vec<u8> = t.iter().copied().collect();
            assert_eq!(back_to_vec, vec);
        }
    }
    #[test]
    fn test_randomly() {
        #![allow(clippy::eq_op)]
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        type Rank = usize;
        type Depth = usize;
        type TestValueType = u16;
        type TestAssociativePositionalList = AssociativePositionalList<TestValueType>;

        fn check_all(test_me: &TestAssociativePositionalList, ref_list: &Vec<TestValueType>) {
            test_me.check_consistent();
            let as_vec: Vec<TestValueType> = test_me.iter().collect();
            assert_eq!(&as_vec, ref_list);
        }

        let mut test_me: TestAssociativePositionalList = AssociativePositionalList::new();
        let mut ref_list: Vec<TestValueType> = Vec::new();

        check_all(&test_me, &ref_list);

        let mut rng = StdRng::seed_from_u64(1);
        let test_size: TestValueType = 100;

        // test without items
        assert!(test_me.is_empty());
        assert!(test_me == test_me);
        assert_eq!(test_me, test_me);

        // initially fill the list with some items in random positions
        for k in 1..test_size + 1 {
            let i = rng.gen_range(0..(ref_list.len() + 1) as TestValueType);
            let inserted = test_me.insert(i as usize, k);
            ref_list.insert(i as usize, k);
            assert!(inserted);
            check_all(&test_me, &ref_list);
        }
        assert!(!test_me.is_empty());
        // check all items are present in the places we expect
        for k in 1..test_size + 1 {
            let j = test_me.find(&k);
            assert!(j.is_some());
            assert!(j.unwrap() < ref_list.len());
            assert!(ref_list[j.unwrap()] == k);
        }
        // try adding some items more than once (random positions again)
        for k in 1..10 {
            let i = rng.gen_range(0..(ref_list.len() + 1) as TestValueType);
            let inserted = test_me.insert(i as usize, k);
            assert!(!inserted);
        }
        for k in 1..10 {
            let i = rng.gen_range(0..(ref_list.len() + 1) as TestValueType);
            let inserted = test_me.insert(i as usize, test_size - k);
            assert!(!inserted);
        }
        check_all(&test_me, &ref_list);
        // test equality when some items are present
        assert!(test_me == test_me);
        assert_eq!(test_me, test_me);
        // remove half of the items (chosen from random positions)
        for _ in 1..(test_size / 2) {
            let i = rng.gen_range(0..ref_list.len() as TestValueType);
            test_me.remove(i as usize);
            ref_list.remove(i as usize);
            check_all(&test_me, &ref_list);
        }
        // use a random add/remove test
        for k in (test_size + 1)..(test_size * 10) + 1 {
            if rng.gen_ratio(1, 2) && !ref_list.is_empty() {
                // test removing a random value
                let i: usize = (rng.gen_range(0..ref_list.len() as TestValueType)) as usize;
                let v: &TestValueType = ref_list.get(i).unwrap();

                assert_eq!(test_me.find(v).unwrap(), i);
                ref_list.remove(i);
                test_me.remove(i);
            } else {
                // test adding a random value
                let i: usize = rng.gen_range(0..ref_list.len() + 1);
                ref_list.insert(i, k);
                let inserted = test_me.insert(i, k);
                assert!(inserted);
                let j = test_me.find(&k);
                assert_eq!(j.unwrap(), i);
            }
            check_all(&test_me, &ref_list);
        }
        // remove the rest of the items
        while !ref_list.is_empty() {
            let i: usize = (rng.gen_range(0..ref_list.len() as TestValueType)) as usize;
            ref_list.remove(i);
            test_me.remove(i);
            check_all(&test_me, &ref_list);
        }
        // test without items again
        assert!(test_me.is_empty());
        assert!(test_me == test_me);
        assert_eq!(test_me, test_me);

        // check that the list works the same after clearing:
        // iteration 0: an empty but used state
        // iteration 1: a non-empty state
        // iteration 2: an empty and unused state
        for j in 0..3 {
            if j == 2 {
                test_me = AssociativePositionalList::new();
            }
            test_me.clear();
            ref_list.clear();
            assert!(test_me.is_empty());
            if j == 2 {
                assert_eq!(test_me.data.len(), 0); // empty and never used
            } else {
                assert_eq!(test_me.data.len(), 1); // empty but used
            }
            for k in 1..10 {
                let i = rng.gen_range(0..(ref_list.len() + 1) as TestValueType);
                let inserted = test_me.insert(i as usize, k);
                ref_list.insert(i as usize, k);
                assert!(inserted);
                check_all(&test_me, &ref_list);
            }
        }

        // compare to a different list in various states
        {
            let mut another: TestAssociativePositionalList = AssociativePositionalList::new();
            assert!(test_me != another);
            for (i, x) in test_me.iter().enumerate() {
                another.insert(i, x);
            }
            assert!(test_me == another); // the other list has the same values
            let v = another[1];
            another.remove(1);
            assert!(test_me != another); // the other list has a different length
            another.insert(1, 0);
            assert!(test_me != another); // the other list has a different value
            another.insert(1, v);
            another.remove(2);
            assert!(test_me == another); // the other list has the same values again
        }
    }

    #[test]
    fn test_interfaces() {
        let mut p: AssociativePositionalList<String> = AssociativePositionalList::new();
        p.insert(0, "Hello".to_string());
        p.insert(1, "World".to_string());
        assert_eq!(p.find(&"World".to_string()), Some(1));
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], "Hello");
        assert_eq!(p[1], "World");
        assert_eq!(&format!("{:?}", p), "[\"Hello\", \"World\"]"); // test Debug formatter
        assert_eq!(p, p);
        assert!(!p.is_empty());
        for n in p.iter() {
            assert!(n == "Hello" || n == "World");
        }
        p.remove(0);
        assert_eq!(p[0], "World");
        assert_eq!(p.find(&"Hello".to_string()), None);
        assert_eq!(p.find(&"World".to_string()), Some(0));
        p.remove(0);
        assert!(p.is_empty());
        assert_eq!(&format!("{:?}", p), "[]");
        let mut p2: AssociativePositionalList<i8> = AssociativePositionalList::new();
        for i in 0..5 {
            p2.insert(0, i);
        }
        assert_eq!(&format!("{:?}", p2), "[4, 3, 2, 1, 0]");
        assert_eq!(p2.find(&0), Some(4));
        p2.remove(1);
        assert_eq!(p2.find(&0), Some(3));
        assert_eq!(&format!("{:?}", p2), "[4, 2, 1, 0]");
    }
}
