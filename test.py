import pyGOURGS
import unittest

class TestNumberBaseConversions(unittest.TestCase):

    def test_base_one(self):
        self.assertEqual(pyGOURGS.decimal_to_base_m(5,1), [1,1,1,1,1])
        self.assertEqual(pyGOURGS.base_m_to_decimal(11111,1), 5)
        self.assertEqual(pyGOURGS.base_m_to_decimal([1,1,1,1,1],1), 5)

    def test_base_two(self):
        self.assertEqual(pyGOURGS.decimal_to_base_m(5,2), [1,0,1])
        self.assertEqual(pyGOURGS.base_m_to_decimal(101,2), 5)
        self.assertEqual(pyGOURGS.base_m_to_decimal([1,0,1],2), 5)
        
    def test_base_three(self):
        self.assertEqual(pyGOURGS.decimal_to_base_m(125,3), [1,1,1,2,2])
        self.assertEqual(pyGOURGS.base_m_to_decimal(11122,3), 125)
        self.assertEqual(pyGOURGS.base_m_to_decimal([1,1,1,2,2],3), 125)
        
    def test_base_nine(self):
        self.assertEqual(pyGOURGS.decimal_to_base_m(125,9), [1,4,8])
        self.assertEqual(pyGOURGS.base_m_to_decimal(148,9), 125)
        self.assertEqual(pyGOURGS.base_m_to_decimal([1,4,8],9), 125)
        
    def test_base_25(self):
        self.assertEqual(125, pyGOURGS.base_m_to_decimal(pyGOURGS.decimal_to_base_m(125,25),25))
        self.assertEqual(183513434438, 
                         pyGOURGS.base_m_to_decimal(pyGOURGS.decimal_to_base_m(
                                                    183513434438,94),94))

class TestITH_NARY_TREE(unittest.TestCase):
    def setUp(self):
        self.pset = pyGOURGS.PrimitiveSet()        
        self.pset.add_operator(add, 2)
        self.pset.add_operator(sub, 2)
        self.pset.add_operator(truediv, 2)
        self.pset.add_variable(1)
        self.enum = pyGOURGS.Enumerator(self.pset)
        
    def test_terminal(self):

        self.assertEqual(self.enum, '.')
        self.assertEqual(pyGOURGS.base_m_to_decimal(11111,1), 5)
        self.assertEqual(pyGOURGS.base_m_to_decimal([1,1,1,1,1],1), 5)

if __name__ == '__main__':
    unittest.main()

