#!/usr/bin/env python

import unittest
import os
from array import array

from QFramework import TQTaggable, TQSample, TQObservable, TQTreeFormulaObservable
from ROOT import TFile, TTree
from ROOT import vector as std_vector

from QFramework import TQPythonTest

class TQTreeFormulaObservableTest(TQPythonTest):
    """
    Test the TQTreeFormulaObservable.
    """

    def setUp(self):
        """
        Create an external file with a tree (scalar and vector branch) and
        create tags and a sample using the external file.
        """
        # create temporary directory
        super(TQTreeFormulaObservableTest, self).setUp()

        # create ROOT file and empty tree
        file = TFile(
            os.path.join(self.tempdir, "TQTreeFormulaObservable.root"),
            "recreate")
        tree = TTree("NOMINAL", "")
        x = array("d", [3.14])
        tree.Branch("n_taus", x, "n_taus/D")

        vec = std_vector('double')()
        vec.push_back(1.42)
        vec.push_back(2.72)
        tree.Branch("tau_pt", vec)

        tree.Fill()

        x[0] = 42
        vec.clear()
        vec.push_back(125.16)
        tree.Fill()

        tree.Write()
        file.Close()

        # create tags and sample
        self.tags = TQTaggable("a=n_taus,b=tau_pt")
        self.sample = TQSample("sample")
        self.sample.importTags(self.tags)
        self.sample = TQSample("sample")
        self.sample.setTreeLocation(
            os.path.join(self.tempdir, "TQTreeFormulaObservable.root:NOMINAL"))

    def tearDown(self):
        # delete temporary directory
        super(TQTreeFormulaObservableTest, self).tearDown()


    def test_basic_retrieval(self):
        """
        Test a TQTreeFormulaObservable can be created/retrieved via TQObservable.
        """
        # scalar
        obs = TQObservable.getObservable("n_taus", self.tags)
        self.assertTrue(obs)
        
        # vector
        obs = TQObservable.getObservable("VecBranch(tau_pt)", self.tags)
        self.assertTrue(obs)

    def test_observable_type_scalar(self):
        """
        Test that the observable created from an expression with a scalar
        branch is a scalar observable.
        """
        obs = TQObservable.getObservable("n_taus", self.tags)
        self.assertIsInstance(obs, TQTreeFormulaObservable)
        self.assertEqual(obs.getObservableType(), TQObservable.scalar)

    def test_observable_type_vector(self):
         """
         Test that the observable created from an expression with a vector
         branch is a vector observable.
         """
         obs = TQObservable.getObservable("VecBranch(tau_pt)", self.tags)
         self.assertIsInstance(obs, TQTreeFormulaObservable)
         self.assertEqual(obs.getObservableType(), TQObservable.vector)

    def test_observable_expr_scalar(self):
        """
        Test that the observable created from an expression with a scalar
        branch returns the same expression again.
        """
        obs = TQObservable.getObservable("n_taus", self.tags)
        self.assertEqual(obs.getExpression(), "n_taus")

    def test_observable_expr_vector(self):
        """
        Test that the observable created from an expression with a vector
        branch returns the same expression again.
        """
        obs = TQObservable.getObservable("VecBranch(tau_pt)", self.tags)
        self.assertEqual(obs.getExpression(), "VecBranch(tau_pt)")
        
    def test_return_value_scalar(self):
        """
        Test that the observable created from a scalar branch returns the
        correct value.
        """
        obs = TQObservable.getObservable("n_taus", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(obs.getNevaluations(), 1)
        self.assertAlmostEqual(obs.getValue(), 3.14)

        # These test fail. There seems to be an issue with PyROOT. Getting an
        # entry in python, does not notify the TTreeFormulas.
        # tree.GetEntry(1)
        # self.assertEqual(obs.getNevaluations(), 1)
        # self.assertAlmostEqual(obs.getValue(), 42)    # is 3.14
        
    def test_return_value_vector(self):
        """
        Test that the observable created from a vector branch returns the
        correct value.
        """
        obs = TQObservable.getObservable("VecBranch(tau_pt)", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(obs.getNevaluations(), 2)
        self.assertAlmostEqual(obs.getValueAt(0), 1.42)
        self.assertAlmostEqual(obs.getValueAt(1), 2.72)

        # These test fail. There seems to be an issue with PyROOT. Getting an
        # entry in python, does not notify the TTreeFormulas.
        # tree.GetEntry(1)
        # self.assertEqual(obs.getNevaluations(), 1)          # is 2
        # self.assertAlmostEqual(obs.getValueAt(0), 125.16)   # is 3.14

    def test_multiple_retrieval_vector(self):
        """
        Test a TQTreeFormulaObservable can be created/retrieved via TQObservable
        and retrieved again with exactly the same expression.
        """
        o1 = TQObservable.getObservable("VecBranch(tau_pt)", self.tags)
        self.assertTrue(o1)
        self.assertTrue(o1.initialize(self.sample))
        
        o2 = TQObservable.getObservable("VecBranch(tau_pt)", self.tags)
        self.assertTrue(o2)
        self.assertTrue(o2.initialize(self.sample))

        self.assertEqual(o1, o2)

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(o1.getNevaluations(), 2)
        self.assertAlmostEqual(o1.getValueAt(0), 1.42)
        self.assertAlmostEqual(o1.getValueAt(1), 2.72)

        self.assertEqual(o2.getNevaluations(), 2)
        self.assertAlmostEqual(o2.getValueAt(0), 1.42)
        self.assertAlmostEqual(o2.getValueAt(1), 2.72)

    def test_multiple_retrieval_scalar(self):
        """
        Test a TQTreeFormulaObservable can be created/retrieved via TQObservable
        and retrieved again with exactly the same expression.
        """
        o1 = TQObservable.getObservable("n_taus", self.tags)
        self.assertTrue(o1)
        self.assertTrue(o1.initialize(self.sample))
        
        o2 = TQObservable.getObservable("n_taus", self.tags)
        self.assertTrue(o2)
        self.assertTrue(o2.initialize(self.sample))

        self.assertEqual(o1, o2)

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(o2.getNevaluations(), 1)
        self.assertAlmostEqual(o2.getValue(), 3.14)

        self.assertEqual(o1.getNevaluations(), 1)
        self.assertAlmostEqual(o1.getValue(), 3.14)

    def test_return_value_vector_at(self):
        """
        Test the VecAT observable works with VecBranch.
        """
        obs = TQObservable.getObservable("VecAT(VecBranch(tau_pt), 1)", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertAlmostEqual(obs.getValue(), 2.72)

    def test_return_value_vector_at_by_10_inside_VecBranch(self):
        """
        Test the VecAT observable works with VecBranch, when the branch is
        divided by 10.
        """
        obs = TQObservable.getObservable("VecAT(VecBranch(tau_pt / 10), 1)", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertAlmostEqual(obs.getValue(), 0.272)

    def test_return_value_vector_at_by_10_inside_VecAT(self):
        """
        Test the VecAT observable works with VecBranch, when the vector
        observable is divided by 10.
        """
        obs = TQObservable.getObservable("VecAT([VecBranch(tau_pt)] / 10, 1)", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertAlmostEqual(obs.getValue(), 0.272)

    def test_return_value_vector_at_by_10_outside(self):
        """
        Test the VecAT observable works with VecBranch, when VecAt is divided
        by 10.
        """
        obs = TQObservable.getObservable("[VecAT(VecBranch(tau_pt), 1)] / 10", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertAlmostEqual(obs.getValue(), 0.272)

    def test_return_value_scalar_tags(self):
        """
        Test that the observable created from a scalar branch using tags
        returns the correct value.
        """
        obs = TQObservable.getObservable("$(a)", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(obs.getNevaluations(), 1)
        self.assertAlmostEqual(obs.getValue(), 3.14)

        # These test fail. There seems to be an issue with PyROOT. Getting an
        # entry in python, does not notify the TTreeFormulas.
        # tree.GetEntry(1)
        # self.assertEqual(obs.getNevaluations(), 1)
        # self.assertAlmostEqual(obs.getValue(), 42)    # is 3.14
        
    def test_return_value_vector_tags(self):
        """
        Test that the observable created from a vector branch using tags
        returns the correct value.
        """
        obs = TQObservable.getObservable("VecBranch($(b))", self.tags)
        self.assertTrue(obs.initialize(self.sample))

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(obs.getNevaluations(), 2)
        self.assertAlmostEqual(obs.getValueAt(0), 1.42)
        self.assertAlmostEqual(obs.getValueAt(1), 2.72)

    def test_multiple_retrieval_vector_tags(self):
        """
        Test a TQTreeFormulaObservable can be created/retrieved via TQObservable
        and retrieved again with exactly the same expression when tags are
        involved.
        """
        o1 = TQObservable.getObservable("VecBranch($(b))", self.tags)
        self.assertTrue(o1)
        self.assertTrue(o1.initialize(self.sample))
        
        o2 = TQObservable.getObservable("VecBranch($(b))", self.tags)
        self.assertTrue(o2)
        self.assertTrue(o2.initialize(self.sample))

        o3 = TQObservable.getObservable("VecBranch(tau_pt)", self.tags)
        self.assertTrue(o3)
        self.assertTrue(o3.initialize(self.sample))

        self.assertEqual(o1, o2)
        self.assertEqual(o1, o3)

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(o1.getNevaluations(), 2)
        self.assertAlmostEqual(o1.getValueAt(0), 1.42)
        self.assertAlmostEqual(o1.getValueAt(1), 2.72)

        self.assertEqual(o2.getNevaluations(), 2)
        self.assertAlmostEqual(o2.getValueAt(0), 1.42)
        self.assertAlmostEqual(o2.getValueAt(1), 2.72)

        self.assertEqual(o3.getNevaluations(), 2)
        self.assertAlmostEqual(o3.getValueAt(0), 1.42)
        self.assertAlmostEqual(o3.getValueAt(1), 2.72)

    def test_multiple_retrieval_scalar_tags(self):
        """
        Test a TQTreeFormulaObservable can be created/retrieved via TQObservable
        and retrieved again with exactly the same expression when tags are
        involved.
        """
        o1 = TQObservable.getObservable("$(a)", self.tags)
        self.assertTrue(o1)
        self.assertTrue(o1.initialize(self.sample))
        
        o2 = TQObservable.getObservable("$(a)", self.tags)
        self.assertTrue(o2)
        self.assertTrue(o2.initialize(self.sample))

        o3 = TQObservable.getObservable("n_taus", self.tags)
        self.assertTrue(o3)
        self.assertTrue(o3.initialize(self.sample))

        self.assertEqual(o1, o2)
        self.assertEqual(o1, o3)

        tree = self.sample.getTreeToken().getContentAsTObject()

        tree.GetEntry(0)
        self.assertEqual(o2.getNevaluations(), 1)
        self.assertAlmostEqual(o2.getValue(), 3.14)

        self.assertEqual(o1.getNevaluations(), 1)
        self.assertAlmostEqual(o1.getValue(), 3.14)

        self.assertEqual(o2.getNevaluations(), 1)
        self.assertAlmostEqual(o2.getValue(), 3.14)
