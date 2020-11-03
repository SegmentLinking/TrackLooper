#!/usr/bin/env python

import unittest

from QFramework import TQTaggable

class TQTaggableTest(unittest.TestCase):
	
	def test_default_constructor(self):
		"""
		Check that the default constructor returns a non-null object and does
		not throw an exception.
		"""
		try:
			tags = TQTaggable()
		except Exception as e:
			self.fail("Default constructor raise an exception: %s" % e)
		
		self.assertTrue(tags)

	def test_constructor_with_tag_string(self):
		"""
		Check that when a TQTaggable is created with a comma separated list of
		tags, the tags retrievable.
		"""
		tags = TQTaggable("tagA=1,tagB=2,tagC=hello")

		# check for existing tags
		self.assertEqual(tags.getTagIntegerDefault("tagA", 0), 1)
		self.assertEqual(tags.getTagIntegerDefault("tagB", 0), 2)
		self.assertEqual(tags.getTagStringDefault("tagC", ""), "hello")

		# check that no other tags exists
		self.assertEqual(tags.getTagIntegerDefault("tagD", 42), 42)
		self.assertEqual(tags.getTagStringDefault("tagC", ""), "hello")


