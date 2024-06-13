import unittest

from unittest.mock import patch, mock_open
from AxfcMachineDesc import AxfcMachineDesc
from AxfcError import AxfcError


class TestAxfcMachineDesc(unittest.TestCase):
    mock_data = """
        {
            "AIX_MODEL_TYPE": "ONNX", 
            "AIX_MODEL_NAME": "test_model", 
            "AIX_PROFIT_THRESHOLD": 1000, 
            "AIX_LAYER": {
                "conv": {
                    "layer": 1, 
                    "activation": 1, 
                    "is_group": false, 
                    "is_conv": true, 
                    "profit": 1500
                }
            }, 
            "STOP_COMPILING_POINT": "node_1", 
            "INPUT_POINT": "input_node"
        }
        """
    
    def setUp(self):
        self.md = AxfcMachineDesc()


    @patch('builtins.open', new_callable=mock_open, read_data=mock_data)
    def test_read_file_success(self, mock_file):
        # States
        result = self.md.read_file("__path.md")

        self.assertEqual(result, AxfcError.SUCCESS)
        self.assertEqual(self.md.get_model_type(), AxfcMachineDesc.TYPE_ONNX)
        self.assertEqual(self.md.get_model_name(), "test_model")
        self.assertEqual(self.md.get_profit_threshold(), 1000)
        self.assertEqual(self.md.get_break_point_node(), 'node_1')
        self.assertEqual(self.md.get_input_point(), "input_node")

    
    @patch("builtins.open", new_callable=mock_open, read_data='{"AIX_MODEL_TYPES" : "ONNX", "AIX_LAYER"  {}}') # Incorrect format (missing :)
    def test_read_file_invalid_format(self, mock_file):
        result = self.md.read_file("dummy_path")
        self.assertEqual(result, AxfcError.INVALID_MD_FORMAT)


    @patch("builtins.open", new_callable=mock_open, read_data='{}')
    def test_read_file_missing_keys(self, mock_file):
        result = self.md.read_file("dummy_path")
        self.assertEqual(result, AxfcError.INVALID_MD_FORMAT)


    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_read_file_not_found(self, mock_file):
        result = self.md.read_file("dummy_path")
        self.assertEqual(result, AxfcError.INVALID_FILE_PATH)


    def test_get_layer_info(self):
        self.md._AxfcMachineDesc__aix_layer_info_tbl = {
            "conv": AxfcMachineDesc.AIXLayerInfo("conv")
        }
        layer_info = self.md.get_layer_info("conv")
        self.assertIsNotNone(layer_info)
        self.assertEqual(layer_info.op, "conv")

        layer_info = self.md.get_layer_info("nonexistent")
        self.assertIsNone(layer_info)

    def test_get_aixh_support(self):
        self.md._AxfcMachineDesc__aix_layer_info_tbl = {
            "conv": AxfcMachineDesc.AIXLayerInfo("conv")
        }
        self.assertTrue(self.md.get_aixh_support("conv"))
        self.assertFalse(self.md.get_aixh_support("nonexistent"))


    def test_get_model_type(self):
        self.md._AxfcMachineDesc__aix_model_info_tbl = {
            "AIX_MODEL_TYPE": "TENSORFLOW"
        }
        self.assertEqual(self.md.get_model_type(), AxfcMachineDesc.TYPE_TENSORFLOW)

        self.md._AxfcMachineDesc__aix_model_info_tbl = {
            "AIX_MODEL_TYPE": "UNKNOWN"
        }
        self.assertEqual(self.md.get_model_type(), AxfcMachineDesc.TYPE_UNKNOWN)


    def test_get_model_name(self):
        self.md._AxfcMachineDesc__aix_model_info_tbl = {
            "AIX_MODEL_NAME": "test_model"
        }
        self.assertEqual(self.md.get_model_name(), "test_model")

        self.md._AxfcMachineDesc__aix_model_info_tbl = {}
        self.assertEqual(self.md.get_model_name(), "NO_NAME")


    def test_get_profit_threshold(self):
        self.md._AxfcMachineDesc__aix_model_info_tbl = {
            "AIX_PROFIT_THRESHOLD": 3000
        }
        self.assertEqual(self.md.get_profit_threshold(), 3000)

        self.md._AxfcMachineDesc__aix_model_info_tbl = {}
        self.assertEqual(self.md.get_profit_threshold(), AxfcMachineDesc.DEFAULT_PROFIT_THRESHOLD)


    def test_get_break_point_node(self):
        self.md._AxfcMachineDesc__aix_model_info_tbl = {
            "STOP_COMPILING_POINT": "node_1"
        }
        self.assertEqual(self.md.get_break_point_node(), "node_1")

        self.md._AxfcMachineDesc__aix_model_info_tbl = {}
        self.assertEqual(self.md.get_break_point_node(), "")


    def test_get_input_point(self):
        self.md._AxfcMachineDesc__aix_model_info_tbl = {
            "INPUT_POINT": "input_node"
        }
        self.assertEqual(self.md.get_input_point(), "input_node")

        self.md._AxfcMachineDesc__aix_model_info_tbl = {}
        self.assertEqual(self.md.get_input_point(), "")


if __name__ == '__main__':
    unittest.main()
