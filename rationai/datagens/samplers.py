"""
TODO: Missing docstring.
"""
# Standard Imports
from abc import abstractmethod
import logging
from dataclasses import dataclass
from typing import List
from typing import Optional

# Third-party Imports
import numpy as np
from numpy.random.mtrand import sample

# Local Imports
from rationai.datagens.datasources import DataSource
from rationai.utils.config import ConfigProto

log = logging.getLogger('samplers')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname).1s][%(process)d][%(filename)s][%(funcName)-25.25s] %(message)s',
                    datefmt='%d.%m.%Y %H:%M:%S')


@dataclass
class SampledEntry:
    """SampledEntry is a dataclass holding a single entry from the dataset
    and metadata information regarding the origin of the entry.

    The metadata is information that may be necessary for extractor to fully
    interpret the sampled data.
    """
    entry: dict
    metadata: dict


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

    def split(self, col: str):
        """Creates a new level of a SamplingTree. Each node receives number
        of new children equal to unique values in the selected column.

        Args:
            col (str): Column name.
        """
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

    def split_node(self, col: str):
        """Leaf node is replaced with an internal node and N new leaf nodes,
        where N is the number of unique values for column `col`.

        The dataframe in the original leaf node is split in such a way that
        there is a single unique value in the split column in each of the new
        leaf node. The values accross leaf nodes all are different.

        Args:
            col (str): Column name.
        """
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
        return f'Node({self.node_name})'


class TreeSampler:
    """TreeSampler is a sampler that utilizes SamplingTree data structure."""

    def __init__(self, config: ConfigProto, data_source: DataSource):
        self.config = config
        self.data_source = data_source
        self.sampling_tree = self.__build_sampling_tree()

    def __build_sampling_tree(self) -> SamplingTree:
        """Builds a SamplingTree from the provided DataSource.

        Function requires that the DataSource.data is a list of paths to DataFrame objects.

        Args:
            data_source (DataSource): DataSource containing paths to input files
            index_levels (List[str]): List of column names appearing in the input DataFrames.
                                      These column names are then used to multi-level sampling
                                      tree. Column names are processed in order of appearance.

        Returns:
            SamplingTree: SamplingTree data structure.
        """
        df = self.data_source.get_table()
        sampling_tree = SamplingTree(df)
        for index_level in self.config.index_levels:
            sampling_tree.split(index_level)
        return sampling_tree

    @abstractmethod
    def sample(self) -> List[SampledEntry]:
        """Defines sampling strategy for a TreeSampler"""
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self) -> List[SampledEntry]:
        """Defines behaviour at the end of an epoch. Typically resampling."""
        raise NotImplementedError


class RandomTreeSampler(TreeSampler):
    """
        RandomSampler samples randomly 'epoch_size' entries.
        Supports multi-level sampling by including 'index_level'.
    """

    def __init__(self, config: ConfigProto, data_source: DataSource):
        super().__init__(config, data_source)

    def sample(self) -> List[SampledEntry]:
        """Returns a list of sampled entries of size equal to `RandomTreeSampler.size`.

        At every node a child branch is chosen uniformly from all children of the current node.

        Returns:
            List[SampledEntry]: [description]
        """
        result = []
        for _ in range(self.config.epoch_size):
            node = self.sampling_tree.root
            while node.children:
                idx = np.random.randint(low=0, high=len(node.children))
                node = node.children[idx]

            entry = node.data.sample().to_dict('records')[0]
            metadata = self.data_source.get_metadata(entry)

            sampled_entry = SampledEntry(
                entry=entry,
                metadata=metadata
            )
            result.append(sampled_entry)
        return result

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.epoch_size = None
            self.index_levels = None

        def parse(self):
            self.epoch_size = self.config.get('epoch_size', None)
            self.index_levels = self.config.get('index_levels', list())


class SequentialTreeSampler(TreeSampler):
    """
        SequentialSampler traverses all leaves once and returns their data content.
        Supports multi-level sampling by including 'index_level'.
    """

    def __init__(self, config: ConfigProto, data_source: DataSource):
        super().__init__(config, data_source)
        self.active_node = self.sampling_tree.leaf

    def sample(self) -> Optional[List[SampledEntry]]:
        """Returns the content of currently active SamplerTree node.

        Returns:
            Optional[List[SampledEntry]]: List of sampled entries.
        """
        if self.active_node is not None:
            result = []
            for entry in self.active_node.data.to_dict('records'):
                metadata = self.data_source.get_metadata(entry)
                sampled_entry = SampledEntry(
                    entry=entry,
                    metadata=metadata
                )
                result.append(sampled_entry)
            return result
        return None

    def next(self) -> None:
        """Sets next leaf as an active node.
        """
        if self.active_node is not None:
            self.active_node = self.active_node.next

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.index_levels = None

        def parse(self):
            self.index_levels = self.config.get('index_levels', list())
