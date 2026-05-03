import unittest
import numpy as np
import os

import data_factory

class TestDataFactory(unittest.TestCase):
    def test_find_input_label_column(self):
        # Assuming the function find_input_label_column is defined in the same file or imported
        path1 = os.path.join("template", "input_label_map_disturbance_force_label.yaml")
        # input_columns, label_columns = data_factory.DataFactory.find_input_label_column(path1, path2)
        input_label_map = data_factory.DaimlDataFactory.get_map(path1)
        expected_input_columns = ["position", "q", "v", "omega", "f_ctrl_input", "torque_ctrl_input", "rotor_0_rotation_spd", "rotor_1_rotation_spd", "rotor_2_rotation_spd", "rotor_3_rotation_spd"]
        expected_label_columns = ["f_disturb"]     
        print(input_label_map["input"].keys())   
        self.assertEqual(set(input_label_map["input"].keys()), set(expected_input_columns))
        self.assertEqual(set(input_label_map["label"].keys()), set(expected_label_columns))

if __name__ == '__main__':
    unittest.main()