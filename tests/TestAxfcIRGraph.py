import unittest

from AxfcIRGraph import AxfcIRGraph
from unittest.mock import Mock, patch
from AxfcError import AxfcError


class TestAxfcIRGraph(unittest.TestCase):


    def setUp(self):
        self.symtab = {}
        self.ir_graph = AxfcIRGraph(self.symtab)
    

    def test_initialization(self):
        self.assertEqual(self.ir_graph.root_node, None)
        self.assertEqual(self.ir_graph.nodes, [])
        self.assertEqual(self.ir_graph.blocks, [])
        self.assertEqual(self.ir_graph.symtab, self.symtab)


    def test_append_node(self):
        node = Mock()
        self.ir_graph.append_node(node)
        self.assertEqual(self.ir_graph.nodes[0], node)
        self.assertEqual(self.ir_graph.root_node, node)
        self.assertEqual(node.id, 0)

        node2 = Mock()
        self.ir_graph.append_node(node2)
        self.assertEqual(self.ir_graph.nodes[1], node2)
        self.assertEqual(node2.id, 1)


    def test_append_block(self):
        block = Mock()
        self.ir_graph.append_block(block)
        self.assertEqual(self.ir_graph.blocks[0], block)
        self.assertEqual(block.id, 0)

        block2 = Mock()
        self.ir_graph.append_block(block2)
        self.assertEqual(self.ir_graph.blocks[1], block2)
        self.assertEqual(block2.id, 1)

    
    def test_analyse_liveness(self):
        block = Mock()
        block.analyse_liveness.return_value = AxfcError.SUCCESS
        self.ir_graph.append_block(block)

        block2 = Mock()
        block2.analyse_liveness.return_value = AxfcError.SUCCESS
        self.ir_graph.append_block(block2)

        self.assertEqual(self.ir_graph.analyse_liveness(), AxfcError.SUCCESS)

        block2.analyse_liveness.return_value = AxfcError.EMPTY_IR_BLOCK
        self.assertEqual(self.ir_graph.analyse_liveness(), AxfcError.EMPTY_IR_BLOCK)

        block2.analyse_liveness.return_value = AxfcError.NOT_AIXH_SUPPORT
        self.assertEqual(self.ir_graph.analyse_liveness(), AxfcError.NOT_AIXH_SUPPORT)

  
if __name__ == '__main__':
    unittest.main()