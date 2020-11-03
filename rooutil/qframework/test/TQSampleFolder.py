#!/usr/bin/env python

import os
import tempfile
import unittest

import ROOT

import QFramework
from QFramework import TQSampleFolder, TQSample, TQCounter

from QFramework import TQPythonTest

class TQSampleFolderTest(TQPythonTest):

    def setUp(self):
        """
        Create temporary test directory.
        """
        # create temporary directory
        super(TQSampleFolderTest, self).setUp()
        self.YESTERDAY = 1495723285
        self.TODAY = 1495809685
    
    def tearDown(self):
        # delete temporary directory
        super(TQSampleFolderTest, self).tearDown()
        pass


    def test_dates(self):
        """
        Ensure that the preset dates YESTERDAY and TODAY are in chronological
        order.
        """
        self.assertLess(self.YESTERDAY, self.TODAY)

    def test_merge_sample_nosum(self):
        """
        Merge two samples without the sum option, this means merging should
        not take place at all, since there are no timestamps.
        """
        this = TQSampleFolder("this")
        s1 = TQSample("htautau")
        s1.addObject(TQCounter("c", 1.234, 35.1, 1234))
        s1.setTag("test.s1", "hello world")
        this.addSampleFolder(s1)

        other = TQSampleFolder("other")
        s2 = TQSample("htautau")
        s2.addObject(TQCounter("c", 4.321, 65.7, 4321))
        s2.setTag("test.s2", "hello multiverse")
        other.addSampleFolder(s2)

        this.merge(other, False)

        S = this.getSample("htautau")
        self.assertTrue(S)
        self.assertIsInstance(S, TQSample)

        C = S.getObject("c")
        self.assertTrue(C)
        self.assertIsInstance(C, TQCounter)
        self.assertEqual(C.getCounter(), 1.234)
        self.assertEqual(C.getRawCounter(), 1234)
        self.assertEqual(C.getError(), 35.1)

    def test_merge_sample_sum(self):
        """
        Merge two samples with the sum option true. In this case time samples
        do no play any role.
        """
        this = TQSampleFolder("this")
        s1 = TQSample("htautau")
        s1.addObject(TQCounter("c", 1.234, 35.1, 1234))
        s1.setTag("test.s1", "hello world")
        this.addSampleFolder(s1)

        other = TQSampleFolder("other")
        s2 = TQSample("htautau")
        s2.addObject(TQCounter("c", 4.321, 65.7, 4321))
        s2.setTag("test.s2", "hello multiverse")
        other.addSampleFolder(s2)

        this.merge(other, "asv", True)

        S = this.getSample("htautau")
        self.assertTrue(S)
        self.assertIsInstance(S, TQSample)
        self.assertEqual(repr(S),
            'TQSample("htautau") @ this:/htautau test.s1 = "hello world", test.s2 = "hello multiverse"')

        C = S.getObject("c")
        self.assertTrue(C)
        self.assertIsInstance(C, TQCounter)
        self.assertEqual(C.getCounter(), 5.555)
        self.assertEqual(C.getRawCounter(), 5555)
        self.assertEqual(C.getError(), 74.48825410761083)


    def test_merge_sample_other_timestamp_nosum(self):
        """
        Merge two samples without the sum option, but the other sample folder
        has a timestamp. This means the other one should be taken.
        """
        this = TQSampleFolder("this")
        s1 = TQSample("htautau")
        s1.addObject(TQCounter("c", 1.234, 35.1, 1234))
        s1.setTag("test.s1", "hello world")
        this.addSampleFolder(s1)

        other = TQSampleFolder("other")
        s2 = TQSample("htautau")
        s2.addObject(TQCounter("c", 4.321, 65.7, 4321))
        s2.setTag("test.s2", "hello multiverse")
        s2.setTag(".test.timestamp.machine", self.YESTERDAY)
        other.addSampleFolder(s2)

        this.merge(other, "test", False)

        S = this.getSample("htautau")
        self.assertTrue(S)
        self.assertIsInstance(S, TQSample)

        C = S.getObject("c")
        self.assertTrue(C)
        self.assertIsInstance(C, TQCounter)
        self.assertEqual(C.getCounter(), 4.321)
        self.assertEqual(C.getRawCounter(), 4321)
        self.assertEqual(C.getError(), 65.7)


    def test_merge_sample_other_older_nosum(self):
        """
        Merge two samples without the sum option, but the other sample folder
        is older. This means this one should be taken.
        """
        this = TQSampleFolder("this")
        s1 = TQSample("htautau")
        s1.addObject(TQCounter("c", 1.234, 35.1, 1234))
        s1.setTag("test.s1", "hello world")
        s1.setTag(".test.timestamp.machine", self.TODAY)
        this.addSampleFolder(s1)

        other = TQSampleFolder("other")
        s2 = TQSample("htautau")
        s2.addObject(TQCounter("c", 4.321, 65.7, 4321))
        s2.setTag("test.s2", "hello multiverse")
        s2.setTag(".test.timestamp.machine", self.YESTERDAY)
        other.addSampleFolder(s2)

        this.merge(other, "test", False)

        S = this.getSample("htautau")
        self.assertTrue(S)
        self.assertIsInstance(S, TQSample)

        C = S.getObject("c")
        self.assertTrue(C)
        self.assertIsInstance(C, TQCounter)
        self.assertEqual(C.getCounter(), 1.234)
        self.assertEqual(C.getRawCounter(), 1234)
        self.assertEqual(C.getError(), 35.1)

    def test_merge_sample_other_newer_nosum(self):
        """
        Merge two samples without the sum option, but the other sample folder
        is newer. This means this one should be taken.
        """
        this = TQSampleFolder("this")
        s1 = TQSample("htautau")
        s1.addObject(TQCounter("c", 1.234, 35.1, 1234))
        s1.setTag("test.s1", "hello world")
        s1.setTag(".test.timestamp.machine", self.YESTERDAY)
        this.addSampleFolder(s1)

        other = TQSampleFolder("other")
        s2 = TQSample("htautau")
        s2.addObject(TQCounter("c", 4.321, 65.7, 4321))
        s2.setTag("test.s2", "hello multiverse")
        s2.setTag(".test.timestamp.machine", self.TODAY)
        other.addSampleFolder(s2)

        this.merge(other, "test", False)

        S = this.getSample("htautau")
        self.assertTrue(S)
        self.assertIsInstance(S, TQSample)

        C = S.getObject("c")
        self.assertTrue(C)
        self.assertIsInstance(C, TQCounter)
        self.assertEqual(C.getCounter(), 4.321)
        self.assertEqual(C.getRawCounter(), 4321)
        self.assertEqual(C.getError(), 65.7)



    def test_merge_mixed(self):
        """
        Test merging with recursive samples, sample folders and folders.
        """
        pass
