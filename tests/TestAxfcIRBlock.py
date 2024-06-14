import unittest

from unittest.mock import Mock
from AxfcIRBlock import AxfcIRBlock
from AxfcError import AxfcError


class TestAxfcIRBlock(unittest.TestCase):

    def setUp(self):
        self.ir_block = AxfcIRBlock()

    def test_initialization(self):
        self.assertEqual(self.ir_block.id, 0)
        self.assertEqual(self.ir_block.nodes, [])
        self.assertEqual(self.ir_block.live_in, set())
        self.assertEqual(self.ir_block.live_out, set())
        self.assertFalse(self.ir_block.is_aixh_support)
        self.assertEqual(self.ir_block.aixh_profit, 0)
        self.assertIsNone(self.ir_block.aix_graph)
        self.assertEqual(self.ir_block.input_nodes, [])
        self.assertIsNone(self.ir_block.output_node)


    def test_analyse_liveness_empty_nodes(self):
        self.ir_block.nodes = None
        result = self.ir_block.analyse_liveness()
        self.assertEqual(result, AxfcError.EMPTY_IR_BLOCK)


    def test_analyse_liveness(self):
        node1 = Mock()
        node1.id = 1
        node1.preds = []
        node1.succs = []
        node1.is_aixh_support = False

        node2 = Mock()
        node2.id = 2
        node2.preds = [node1]
        node2.succs = []
        node2.is_aixh_support = False

        self.ir_block.nodes = [node1, node2]
        result = self.ir_block.analyse_liveness()

        self.assertEqual(result, AxfcError.SUCCESS)


    def test_analyse_inout_empty_nodes(self):
        self.ir_block.nodes = None
        result = self.ir_block._AxfcIRBlock__analyse_inout()
        self.assertEqual(result, AxfcError.EMPTY_IR_BLOCK)


    def test_analyse_inout(self):
        self.ir_block.nodes = [Mock()]
        result = self.ir_block._AxfcIRBlock__analyse_inout()
        self.assertEqual(result, AxfcError.SUCCESS)


    def test_analyze_profit_empty_nodes(self):
        self.ir_block.nodes = None
        result, profit = self.ir_block.analyze_profit()
        self.assertEqual(result, AxfcError.EMPTY_IR_BLOCK)
        self.assertEqual(profit, -1)


    def test_analyze_profit_not_supported(self):
        self.ir_block.nodes = [Mock()]
        self.ir_block.is_aixh_support = True
        result, profit = self.ir_block.analyze_profit()
        self.assertEqual(result, AxfcError.NOT_AIXH_SUPPORT)
        self.assertEqual(profit, -1)


    def test_analyze_profit(self):
        node1 = Mock()
        node1.op = "Conv"
        node1.analyze_profit.return_value = 10

        node2 = Mock()
        node2.op = "ReLU"
        node2.analyze_profit.return_value = 5

        self.ir_block.nodes = [node1, node2]
        self.ir_block.is_aixh_support = False
        result = self.ir_block.analyze_profit()

        self.assertEqual(result, AxfcError.SUCCESS)
        self.assertEqual(self.ir_block.aixh_profit, 15)


if __name__ == '__main__':
    unittest.main()
