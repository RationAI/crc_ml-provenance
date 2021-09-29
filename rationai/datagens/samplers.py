# Standard Imports
from typing import Optional
from typing import List
import logging

# Third-party Imports
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

log = logging.getLogger('samplers')
logging.basicConfig(level=logging.INFO,
                   format='[%(asctime)s][%(levelname).1s][%(process)d][%(filename)s][%(funcName)-25.25s] %(message)s',
                   datefmt='%d.%m.%Y %H:%M:%S')

class SamplingTree:
    """
        DataStructure for sampling.

        - The root node holds reference to the left-most leaf.
        - The leaves of a SamplingTree are singly linked, allowing
        quick one-way traversal of all leaves.
        - Only leaf nodes of a SamplingTree holds data.
        - On column split, each leaf node introduces G new children nodes, where G
          is the number of unique values in the column. These new nodes form a new
          tree level.
        - The order of columns in which the SamplingTree defines the final form
          of the SamplingTree.
    """
    def __init__(self, df):
        root_node = Node('ROOT', df)
        self.root = root_node
        self.leaf = root_node
        self.split_cols = []

    def split(self, col):
        # Idempotent operation
        if col in self.split_cols:
            return

        # Check if column exists
        cur_node = self.leaf
        assert col in cur_node.data, f'Column {col} does not exist.'

        # Split all leaves and create one new level
        cur_node.split_node(col)
        self.leaf = cur_node.children[0]
        while cur_node.next is not None:
            prev_node = cur_node
            cur_node = cur_node.next
            cur_node.split_node(col)
            prev_node.children[-1].next = cur_node.children[0]
        self.split_cols.append(col)

class Node:
    """
        Node object for SamplingTree.

        Splitting a node on a column with G unique values partitions
        the DataFrame into G new DataFrames. For each new DataFrame a
        child Node is created. Split Node content is erased to free memory.

        All children nodes are chained via 'next' reference to allow
        quick traversal of all leaf nodes.
    """
    def __init__(self, name, df):
        self.node_name = name
        self.data = df
        self.parent = None
        self.children = []
        self.next = None

    def split_node(self, col):
        partitions = {col_val: df for col_val, df in self.data.groupby(col)}
        for col_val, df in partitions.items():
            new_node = Node(col_val, df)
            new_node.parent = self
            self.children.append(new_node)

        for node, next_node in zip(self.children[:-1], self.children[1:]):
            node.next = next_node

        self.data = None
        self.next = None

    def __repr__(self):
        return f'Node({self.name})'

class TreeSampler:
    def __init__(self, data_source, index_levels=[]):
        self.index_levels = index_levels
        self.data_source = data_source
        self.sampling_tree = self.__build_sampling_tree(data_source, index_levels)

    def __build_sampling_tree(self, data_source, index_levels):
        raise NotImplemented()

class RandomSampler(TreeSampler):
    """
        RandomSampler samples randomly 'epoch_size' entries.
        Supports multi-level sampling by including 'index_level'.
    """
    def __init__(self, epoch_size: int, data_source, index_levels: List[str] = []):
        super().__init__(data_source, index_levels)
        self.size = epoch_size

    def sample(self) -> None:
        """Returns randomly sampled entry. At every node a child branch is chosen
           uniformly from all children of the current node.
        """
        result = []
        for _ in range(self.size):
            node = self.sampling_tree.root
            while node.children:
                idx = np.random.randint(low=0, high=len(node.children))
                node = node.children[idx]
            result.append(node.data.sample().to_dict())
        return result

class SequentialSampler(TreeSampler):
    """
        SequentialSampler traverses all leaves once and returns their data content.
        Supports multi-level sampling by including 'index_level'.
    """
    def __init__(self, data_source, index_levels: List[str] = []):
        super().__init__(data_source, index_levels)
        self.active_node = self.sampling_tree.leaf

    def sample(self) -> Optional[List]:
        """Returns the content of currently active SamplerTree node.

        Returns:
            Optional[List]: List of sampled entries.
        """
        if self.active_node is not None:
            return self.active_node.data.to_dict('records')
        return None

    def next(self) -> None:
        """Sets next leaf as an active node.
        """
        if self.active_node is not None:
            self.active_node = self.active_node.next
