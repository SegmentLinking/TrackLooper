#!/usr/bin/env python

import unittest
import os

from QFramework import TQTaggable, TQSample, TQObservable, TQFilterObservable
from ROOT import TFile, TTree

from QFramework import TQPythonTest

class TQFilterObservableTest(TQPythonTest):
	"""
	Test the TQFilterObservable.
	"""

	def setUp(self):
		"""
		Create an external file with a single, empty tree and create tags and
		a sample using the external file.
		"""
		# create temporary directory
		super(TQFilterObservableTest, self).setUp()

		# create ROOT file and empty tree
		file = TFile(
			os.path.join(self.tempdir, "TQFilterObservable.root"),
			"recreate")
		tree = TTree("NOMINAL", "")
		tree.Write()
		file.Close()

		# create tags and sample
		self.tags = TQTaggable("a=2.2,b=3.3")
		self.tags.setTagBool("c", True)
		self.sample = TQSample("sample")
		self.sample.importTags(self.tags)
		self.sample.setTreeLocation(
			os.path.join(self.tempdir, "TQFilterObservable.root:NOMINAL"))

	def tearDown(self):
		# delete temporary directory
		super(TQFilterObservableTest, self).tearDown()
		

	def test_basic_retrieval(self):
		"""
		Test a TQFilterObservable can be created/retrieved via TQObservable.
		"""
		obs = TQObservable.getObservable("Filter($(a), $(b) > 3.)", self.tags)
		self.assertTrue(obs)
		
	def test_return_value(self):
		"""
		Test that the first argument is returned when the second argument is
		true.
		"""
		obs = TQObservable.getObservable("Filter($(a), $(b) > 3.)", self.tags)
		self.assertTrue(obs.initialize(self.sample))

		self.assertEqual(obs.getNevaluations(), 1)
		self.assertEqual(obs.getValueAt(0), 2.2)

	def test_cut(self):
		"""
		Test that the observable does not return any value, when the second
		argument is false.
		"""
		obs = TQObservable.getObservable("Filter($(a), $(b) > 4.)", self.tags)
		self.assertTrue(obs.initialize(self.sample))

		self.assertEqual(obs.getNevaluations(), 0)


	def test_multiple_retrieval(self):
		"""
		Test a TQFilterObservable can be created/retrieved via TQObservable
		and retrieved again with exactly the same expression.
		"""
		o1 = TQObservable.getObservable("Filter($(a), $(b) > 3.)", self.tags)
		self.assertTrue(o1)
		
		o2 = TQObservable.getObservable("Filter($(a), $(b) > 3.)", self.tags)
		self.assertTrue(o2)

		self.assertEqual(o1, o2)

	def test_multiple_retrieval_different_spaces(self):
		"""
		Test a TQFilterObservable can be created/retrieved via TQObservable
		and retrieved again with different number of spaces.
		"""
		o1 = TQObservable.getObservable("Filter($(a), [$(b)] > 3.)", self.tags)
		self.assertTrue(o1)
		
		o2 = TQObservable.getObservable("Filter($(a),[$(b)]> 3.)", self.tags)
		self.assertTrue(o2)
		
		self.assertEqual(o1, o2)
		

	def test_multiple_retrieval_with_bools(self):
		"""
		Test a TQFilterObservable can be created/retrieved via TQObservable
		and retrieved again with exactly the same expression when a boolean
		tag is used in the cut expression.
		"""
		o1 = TQObservable.getObservable("Filter($(a), $(b) > 3. && $(c))", self.tags)
		self.assertTrue(o1)
		
		o2 = TQObservable.getObservable("Filter($(a), $(b) > 3. && $(c))", self.tags)
		self.assertTrue(o2)
		
		self.assertEqual(o1, o2)

	def test_multiple_retrieval_with_bools_and_different_spaces(self):
		"""
		Test a TQFilterObservable can be created/retrieved via TQObservable
		and retrieved again with different usage of white spaces when a boolean
		tag is used in the cut expression.
		"""
		o1 = TQObservable.getObservable("Filter($(a), $(b) > 3. && $(c))", self.tags)
		self.assertTrue(o1)
		
		o2 = TQObservable.getObservable("Filter($(a),$(b)> 3. && $(c))", self.tags)
		self.assertTrue(o2)
		
		self.assertEqual(o1, o2)
		
	
