#!/usr/bin/env python

import os
import tempfile
import unittest

import ROOT

import QFramework
from QFramework import TQFolder

from QFramework import TQPythonTest

class TQFolderTest(TQPythonTest):

    def setUp(self):
        """
        Create temporary test directory.
        """
        # create temporary directory
        super(TQFolderTest, self).setUp()
    
    def tearDown(self):
        # delete temporary directory
        super(TQFolderTest, self).tearDown()
        pass

    def test_merge_disjoint_folders_nosum(self):
        """
        Check the merge results of two disjoint folders with tags, histograms
        and counters without summing. 
        """
        ### prepare
        # prepare 'this'
        this = TQFolder("this")
        a = TQFolder("a")
        a.addFolder(TQFolder("aa"))
        a.addFolder(TQFolder("ab"))
        a.setTag("test.comment", "this is a")

        ha = ROOT.TH1F("ha", "", 10, 0., 1.);
        ha.SetBinContent(1, 42)
        ha.SetBinContent(2, 3.14)
        ha.SetDirectory(0)
        a.addObject(ha)

        ca = QFramework.TQCounter("ca")
        ca.setCounter(1.234)
        ca.setRawCounter(1234)
        ca.setError(35.1)
        a.addObject(ca)

        this.addFolder(a)

        # prepare 'other'
        other = TQFolder("other")
        b = TQFolder("b")
        b.addFolder(TQFolder("ba"))
        bb = TQFolder("bb")
        bb.setTag("test.comment", "this is bb")
        b.addFolder(bb)
        b.setTag("test.comment", "this is b")

        hb = ROOT.TH1F("hb", "", 10, 0., 1.);
        hb.SetBinContent(1, 43)
        hb.SetBinContent(2, 4.14)
        hb.SetDirectory(0)
        b.addObject(hb)

        cb = QFramework.TQCounter("cb")
        cb.setCounter(4.321)
        cb.setRawCounter(4321)
        cb.setError(65.7)
        b.addObject(cb)

        other.addFolder(b)

        ### merge
        this.merge(other)  # sum = False

        ### check result
        # check 'a' sub tree
        A = this.getFolder("a")
        self.assertTrue(A)
        self.assertIsInstance(A, TQFolder)
        self.assertEqual(hash(A), hash(a))
        self.assertEqual(repr(A), "TQFolder(\"a\") @ this:/a test.comment = \"this is a\"")

        HA = A.getObject("ha")
        self.assertTrue(HA)
        self.assertIsInstance(HA, ROOT.TH1F)
        self.assertAlmostEqual(HA.GetBinContent(1), 42, places=5)
        self.assertAlmostEqual(HA.GetBinContent(2), 3.14, places=5)
        self.assertAlmostEqual(HA.GetBinContent(3), 0, places=5)

        CA = A.getObject("ca")
        self.assertTrue(CA)
        self.assertIsInstance(CA, QFramework.TQCounter)
        self.assertAlmostEqual(CA.getCounter(), 1.234)
        self.assertAlmostEqual(CA.getRawCounter(), 1234)
        self.assertAlmostEqual(CA.getError(), 35.1)

        AA = this.getFolder("a/aa")
        self.assertTrue(AA)
        self.assertIsInstance(AA, TQFolder)
        self.assertEqual(repr(AA), "TQFolder(\"aa\") @ this:/a/aa ") 

        AB = this.getFolder("a/ab")
        self.assertTrue(AB)
        self.assertIsInstance(AB, TQFolder)
        self.assertEqual(repr(AB), "TQFolder(\"ab\") @ this:/a/ab ") 

        # check 'b' sub tree
        B = this.getFolder("b")
        self.assertTrue(B)
        self.assertIsInstance(B, TQFolder)
        self.assertEqual(hash(B), hash(b))
        self.assertEqual(repr(B), "TQFolder(\"b\") @ this:/b test.comment = \"this is b\"")

        HB = B.getObject("hb")
        self.assertTrue(HB)
        self.assertIsInstance(HB, ROOT.TH1F)
        self.assertAlmostEqual(HB.GetBinContent(1), 43, places=5)
        self.assertAlmostEqual(HB.GetBinContent(2), 4.14, places=5)
        self.assertAlmostEqual(HB.GetBinContent(3), 0, places=5)

        CB = B.getObject("cb")
        self.assertTrue(CB)
        self.assertIsInstance(CB, QFramework.TQCounter)
        self.assertAlmostEqual(CB.getCounter(), 4.321)
        self.assertAlmostEqual(CB.getRawCounter(), 4321)
        self.assertAlmostEqual(CB.getError(), 65.7)

        BA = this.getFolder("b/ba")
        self.assertTrue(BA)
        self.assertIsInstance(BA, TQFolder)
        self.assertEqual(repr(BA), "TQFolder(\"ba\") @ this:/b/ba ") 

        BB = this.getFolder("b/bb")
        self.assertTrue(BB)
        self.assertIsInstance(BB, TQFolder)
        self.assertEqual(repr(BB), "TQFolder(\"bb\") @ this:/b/bb test.comment = \"this is bb\"") 


    def test_merge_folders_common_base_nosum(self):
        """
        Check the merge results of two folders with common base folders, but
        disjoint with tags, histograms and counters without summing. 
        """
        ### prepare
        # prepare 'this'
        this = TQFolder("this")
        a = TQFolder("a")
        a.addFolder(TQFolder("aa"))
        a.addFolder(TQFolder("ab"))
        a.setTag("test.comment", "this is a")

        ha = ROOT.TH1F("ha", "", 10, 0., 1.);
        ha.SetBinContent(1, 42)
        ha.SetBinContent(2, 3.14)
        ha.SetDirectory(0)
        a.addObject(ha)

        ca = QFramework.TQCounter("ca")
        ca.setCounter(1.234)
        ca.setRawCounter(1234)
        ca.setError(35.1)
        a.addObject(ca)

        this.addFolder(a)

        # prepare 'other'
        other = TQFolder("other")
        b = TQFolder("a")
        b.addFolder(TQFolder("ba"))
        bb = TQFolder("bb")
        bb.setTag("test.comment", "this is bb")
        b.addFolder(bb)
        b.setTag("test.comment2", "this is b")

        hb = ROOT.TH1F("hb", "", 10, 0., 1.);
        hb.SetBinContent(1, 43)
        hb.SetBinContent(2, 4.14)
        hb.SetDirectory(0)
        b.addObject(hb)

        cb = QFramework.TQCounter("cb")
        cb.setCounter(4.321)
        cb.setRawCounter(4321)
        cb.setError(65.7)
        b.addObject(cb)

        other.addFolder(b)

        ### merge
        this.merge(other)  # sum = False

        ### check result
        # check 'a' sub tree
        A = this.getFolder("a")
        self.assertTrue(A)
        self.assertIsInstance(A, TQFolder)
        self.assertEqual(hash(A), hash(a))
        self.assertEqual(repr(A), "TQFolder(\"a\") @ this:/a test.comment = \"this is a\", test.comment2 = \"this is b\"")

        HA = A.getObject("ha")
        self.assertTrue(HA)
        self.assertIsInstance(HA, ROOT.TH1F)
        self.assertAlmostEqual(HA.GetBinContent(1), 42, places=5)
        self.assertAlmostEqual(HA.GetBinContent(2), 3.14, places=5)
        self.assertAlmostEqual(HA.GetBinContent(3), 0, places=5)

        CA = A.getObject("ca")
        self.assertTrue(CA)
        self.assertIsInstance(CA, QFramework.TQCounter)
        self.assertAlmostEqual(CA.getCounter(), 1.234)
        self.assertAlmostEqual(CA.getRawCounter(), 1234)
        self.assertAlmostEqual(CA.getError(), 35.1)

        AA = this.getFolder("a/aa")
        self.assertTrue(AA)
        self.assertIsInstance(AA, TQFolder)
        self.assertEqual(repr(AA), "TQFolder(\"aa\") @ this:/a/aa ") 

        AB = this.getFolder("a/ab")
        self.assertTrue(AB)
        self.assertIsInstance(AB, TQFolder)
        self.assertEqual(repr(AB), "TQFolder(\"ab\") @ this:/a/ab ") 

        # check 'b' sub tree

        HB = A.getObject("hb")
        self.assertTrue(HB)
        self.assertIsInstance(HB, ROOT.TH1F)
        self.assertAlmostEqual(HB.GetBinContent(1), 43, places=5)
        self.assertAlmostEqual(HB.GetBinContent(2), 4.14, places=5)
        self.assertAlmostEqual(HB.GetBinContent(3), 0, places=5)

        CB = A.getObject("cb")
        self.assertTrue(CB)
        self.assertIsInstance(CB, QFramework.TQCounter)
        self.assertAlmostEqual(CB.getCounter(), 4.321)
        self.assertAlmostEqual(CB.getRawCounter(), 4321)
        self.assertAlmostEqual(CB.getError(), 65.7)

        BA = this.getFolder("a/ba")
        self.assertTrue(BA)
        self.assertIsInstance(BA, TQFolder)
        self.assertEqual(repr(BA), "TQFolder(\"ba\") @ this:/a/ba ") 

        BB = this.getFolder("a/bb")
        self.assertTrue(BB)
        self.assertIsInstance(BB, TQFolder)
        self.assertEqual(repr(BB), "TQFolder(\"bb\") @ this:/a/bb test.comment = \"this is bb\"") 



    def test_merge_folders_conflict_histo_nosum(self):
        """
        Check the merge results of two folders with a conflicting histogram,
        without summing.

        The existing histogram should have precedence over the one in the
        other folder.
        """
        ### prepare
        # prepare 'this'
        this = TQFolder("this")
        ha = ROOT.TH1F("ha", "", 10, 0., 1.);
        ha.SetBinContent(1, 42)
        ha.SetBinContent(2, 3.14)
        ha.SetDirectory(0)
        this.addObject(ha)

        # prepare 'other'
        other = TQFolder("other")
        hb = ROOT.TH1F("ha", "", 10, 0., 1.);
        hb.SetBinContent(1, 43)
        hb.SetBinContent(2, 4.14)
        hb.SetDirectory(0)
        other.addObject(hb)

        ### merge
        this.merge(other)  # sum = False

        ### check result
        HA = this.getObject("ha")
        self.assertTrue(HA)
        self.assertIsInstance(HA, ROOT.TH1F)
        self.assertAlmostEqual(HA.GetBinContent(1), 43, places=5)
        self.assertAlmostEqual(HA.GetBinContent(2), 4.14, places=5)
        self.assertAlmostEqual(HA.GetBinContent(3), 0, places=5)

    def test_merge_folders_conflict_histo_sum(self):
        """
        Check the merge results of two folders with a conflicting histogram,
        with summing.

        The histograms should be summed.
        """
        ### prepare
        # prepare 'this'
        this = TQFolder("this")
        ha = ROOT.TH1F("ha", "", 10, 0., 1.);
        ha.SetBinContent(1, 42)
        ha.SetBinContent(2, 3.14)
        ha.SetDirectory(0)
        this.addObject(ha)

        # prepare 'other'
        other = TQFolder("other")
        hb = ROOT.TH1F("ha", "", 10, 0., 1.);
        hb.SetBinContent(1, 43)
        hb.SetBinContent(2, 4.14)
        hb.SetDirectory(0)
        other.addObject(hb)

        ### merge
        this.merge(other, True)  # sum = True

        ### check result
        HA = this.getObject("ha")
        self.assertTrue(HA)
        self.assertIsInstance(HA, ROOT.TH1F)
        self.assertAlmostEqual(HA.GetBinContent(1), 85, places=5)
        self.assertAlmostEqual(HA.GetBinContent(2), 7.28, places=5)
        self.assertAlmostEqual(HA.GetBinContent(3), 0, places=5)

    def test_merge_folders_conflict_counter_nosum(self):
        """
        Check the merge results of two folders with a conflicting counter,
        without summing.

        The other counter should have precedence over the one in the
        this folder.
        """
        ### prepare
        # prepare 'this'
        this = TQFolder("this")
        ca = QFramework.TQCounter("ca")
        ca.setCounter(1.234)
        ca.setRawCounter(1234)
        ca.setError(35.1)
        this.addObject(ca)

        # prepare 'other'
        other = TQFolder("other")
        cb = QFramework.TQCounter("ca")
        cb.setCounter(4.321)
        cb.setRawCounter(4321)
        cb.setError(65.7)
        other.addObject(cb)

        ### merge
        this.merge(other)  # sum = False

        ### check result
        CA = this.getObject("ca")
        self.assertTrue(CA)
        self.assertIsInstance(CA, QFramework.TQCounter)
        self.assertAlmostEqual(CA.getCounter(), 4.321)
        self.assertAlmostEqual(CA.getRawCounter(), 4321)
        self.assertAlmostEqual(CA.getError(), 65.7)

    def test_merge_folders_conflict_counter_sum(self):
        """
        Check the merge results of two folders with a conflicting counter,
        without summing.

        The counters should be added, errors should be added in quadrature.
        """
        ### prepare
        # prepare 'this'
        this = TQFolder("this")
        ca = QFramework.TQCounter("ca")
        ca.setCounter(1.234)
        ca.setRawCounter(1234)
        ca.setError(35.1)
        this.addObject(ca)

        # prepare 'other'
        other = TQFolder("other")
        cb = QFramework.TQCounter("ca")
        cb.setCounter(4.321)
        cb.setRawCounter(4321)
        cb.setError(65.7)
        other.addObject(cb)

        ### merge
        this.merge(other, True)  # sum = True

        ### check result
        CA = this.getObject("ca")
        self.assertTrue(CA)
        self.assertIsInstance(CA, QFramework.TQCounter)
        self.assertAlmostEqual(CA.getCounter(), 5.555)
        self.assertAlmostEqual(CA.getRawCounter(), 5555)
        self.assertAlmostEqual(CA.getError(), 74.48825410761083)
