import pyGOURGS
from pyGOURGS import decimal_to_base_m, base_m_to_decimal
import unittest
from operator import add, sub, truediv, mul


class TestNumberBaseConversions(unittest.TestCase):

    def test_base_one(self):
        self.assertEqual(decimal_to_base_m(5,1), [1,1,1,1,1])
        self.assertEqual(base_m_to_decimal(11111,1), 5)
        self.assertEqual(base_m_to_decimal([1,1,1,1,1],1), 5)

    def test_base_two(self):
        self.assertEqual(decimal_to_base_m(5,2), [1,0,1])
        self.assertEqual(base_m_to_decimal(101,2), 5)
        self.assertEqual(base_m_to_decimal([1,0,1],2), 5)
        
    def test_base_three(self):
        self.assertEqual(decimal_to_base_m(125,3), [1,1,1,2,2])
        self.assertEqual(base_m_to_decimal(11122,3), 125)
        self.assertEqual(base_m_to_decimal([1,1,1,2,2],3), 125)
        
    def test_base_nine(self):
        self.assertEqual(decimal_to_base_m(125,9), [1,4,8])
        self.assertEqual(base_m_to_decimal(148,9), 125)
        self.assertEqual(base_m_to_decimal([1,4,8],9), 125)
        
    def test_base_25(self):
        self.assertEqual(125, base_m_to_decimal(decimal_to_base_m(
                                                125,25),25))
        self.assertEqual(183513434438, base_m_to_decimal(
                            decimal_to_base_m(183513434438,94),94))


class TestSymbolicRegression(unittest.TestCase):

    def setUp(self):
        self.pset = pyGOURGS.PrimitiveSet()
        self.pset.add_operator(add, 2)
        self.pset.add_operator(sub, 1)
        self.pset.add_operator(truediv, 3)
        self.pset.add_variable(1)
        self.enum = pyGOURGS.Enumerator(self.pset)

    def test_count_unique_trees(self):
        trees = list()
        N_trees = 2000
        for i in range(0,N_trees):
            tree = self.enum.ith_n_ary_tree(i)
            trees.append(tree)
            print(tree)
        self.assertEqual(len(list(set(trees))), N_trees)

    def test_terminal(self):
        self.assertEqual(self.enum.ith_n_ary_tree(0), '.')

    def test_operator(self):
        self.assertEqual(self.enum.ith_n_ary_tree(1), '[.]')

    def test_operator(self):
        self.assertEqual(self.enum.ith_n_ary_tree(2), '[.,.]')

    def test_operator(self):
        self.assertEqual(self.enum.ith_n_ary_tree(3), '[.,.,.]')

    def test_count_operators_0(self):
        self.assertEqual(self.enum.calculate_l_i_b(0, 0), 0)

    def test_count_operators_0(self):
        self.assertEqual(self.enum.calculate_l_i_b(1, 0), 1)

    def test_count_operators_0(self):
        self.assertEqual(self.enum.calculate_l_i_b(2, 1), 1)
        self.assertEqual(self.enum.calculate_l_i_b(2, 0), 0)


if __name__ == '__main__':
    unittest.main()

