import unittest

from AxfcIRNode import AxfcIRNode


class TestAxfcIRNode(unittest.TestCase):


    def setUp(self):
        self.node_def = { "key": "value" }
        self.ir_node = AxfcIRNode(self.node_def)


    def test_initialization(self):
        self.assertEqual(self.ir_node.id, 0)
        self.assertEqual(self.ir_node.name, '')
        self.assertEqual(self.ir_node.layer_id, 0)
        self.assertEqual(self.ir_node.succs, [])
        self.assertEqual(self.ir_node.preds, [])
        self.assertEqual(self.ir_node.node_def, self.node_def)
        self.assertIsNone(self.ir_node.block_ref)
        self.assertEqual(self.ir_node.aixh_profit, 0)
        self.assertFalse(self.ir_node.is_aixh_support)
        self.assertFalse(self.ir_node.eval_flag)
        self.assertFalse(self.ir_node.is_input)
        self.assertFalse(self.ir_node.is_output)
        self.assertIsNone(self.ir_node.aix_layer)
        self.assertIsNone(self.ir_node.op)


    def test_analyze_profit(self):
        self.assertEqual(self.ir_node.analyze_profit(), 0)
        self.ir_node.aixh_profit = 100
        self.assertEqual(self.ir_node.analyze_profit(), 100)


    def test_equality(self):
        ir_node2 = AxfcIRNode(self.node_def)
        ir_node2.id = 1
        self.assertNotEqual(self.ir_node, ir_node2)

        ir_node2.id = 0
        self.assertEqual(self.ir_node, ir_node2)


if __name__ == '__main__':
    unittest.main()