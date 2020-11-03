#!/usr/bin/env python

import os
import unittest

from QFramework import TQCut, TQHistoMakerAnalysisJob
from ROOT import TObjArray

class TQCutTest(unittest.TestCase):
    """
    Test the TQCut observable.
    """

    def get_cut_hierarchy(self):
        """
        Helper method to create a toy cut hierarchy.
        """
        base =  TQCut("CutBase")

        trigger = TQCut("CutTrigger")
        base.addCut(trigger)

        OS = TQCut("CutOS")
        trigger.addCut(OS)

        # Signal Region
        BVeto = TQCut("CutBVeto")
        OS.addCut(BVeto)

        PreselectionSR = TQCut("CutPreselectionSR")
        BVeto.addCut(PreselectionSR)

        VbfSR = TQCut("CutVbfSR")
        PreselectionSR.addCut(VbfSR)

        BoostedSR = TQCut("CutBoostedSR")
        PreselectionSR.addCut(BoostedSR)

        # Top Control Region
        BReq = TQCut("CutBReq")
        OS.addCut(BReq)

        PreselectionTopCR = TQCut("CutPreselectionTopCR")
        BReq.addCut(PreselectionTopCR)

        VbfTopCR = TQCut("CutVbfTopCR")
        PreselectionTopCR.addCut(VbfTopCR)

        BoostedTopCR = TQCut("CutBoostedTopCR")
        PreselectionTopCR.addCut(BoostedTopCR)

        return base


    def test_get_cut_hierarchy(self):
        """
        Test the cut hierarchy returned by the get_cut_hierarchy method.
        """
        base = self.get_cut_hierarchy()
        self.assertEqual(base.GetName(), "CutBase")

        trigger = base.getCut("CutTrigger")
        self.assertTrue(trigger)

        OS = trigger.getCut("CutOS")
        self.assertTrue(OS)

        BVeto = OS.getCut("CutBVeto")
        self.assertTrue(BVeto)

        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        self.assertTrue(PreselectionSR)

        VbfSR = PreselectionSR.getCut("CutVbfSR")
        self.assertTrue(VbfSR)

        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        self.assertTrue(BoostedSR)

        BReq = OS.getCut("CutBReq")
        self.assertTrue(BReq)

        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        self.assertTrue(PreselectionTopCR)

        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        self.assertTrue(VbfTopCR)

        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")
        self.assertTrue(BoostedTopCR)

    
    def test_add_analysis_job_singal_cut_name(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to the single cut
        specified by its name. (e.g. "@CutName: histos;") 
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutPreselectionSR")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)

    
    def test_add_analysis_job_list_of_cut_names_wo_spaces(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to all the cuts
        in a comma separated list of cut names.  (e.g. "@CutOne,CutTwo: histos;")  
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutPreselectionSR,CutPreselectionTopCR")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)
    
    def test_add_analysis_job_list_of_cut_names_w_spaces(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to all the cuts
        in a comma separated list of cut names.  (e.g. "@CutOne, CutTwo: histos;")  
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutPreselectionSR, CutPreselectionTopCR")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)
    
    def test_add_analysis_job_cut_name_trailing_asterisk(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to all child cuts
        following the cut before the wildcard.  (e.g. "@CutParent/*: histos;") 
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutBVeto/*")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 1)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 1)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 1)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)

    def test_add_analysis_job_cut_name_trailing_asterisk_omitted_slash(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to all child cuts
        following the cut before the wildcard, when omitting the directory
        slash.  (e.g. "@CutParent*: histos;") 
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutBVeto*")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 1)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)

    def test_get_matching_cuts_single_name(self):
        """
        Test getMatchingCuts when supplied with a single (existing) cut name.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutTrigger")

    def test_get_matching_cuts_single_name_non_existing(self):
        """
        Test getMatchingCuts when supplied with a single (non-existing) cut name.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutNonExisting")
        self.assertEqual(cuts.GetEntries(), 0)

    def test_get_matching_cuts_path_non_wildcards(self):
        """
        Test getMatchingCuts when supplied with a cut path.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutOS/CutBVeto/CutPreselectionSR")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutPreselectionSR")

    def test_get_matching_cuts_path_non_wildcards_non_existing(self):
        """
        Test getMatchingCuts when supplied with a cut path (non-existing).
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutNonExisting/CutOS/CutBVeto/CutPreselectionSR")
        self.assertEqual(cuts.GetEntries(), 0)

    def test_get_matching_cuts_path_with_question_mark(self):
        """
        Test getMatchingCuts when supplied with a cut path with a question
        mark.
        """
        # beginning
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "?/CutOS")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutOS")

        # mid
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutOS/?/CutPreselectionSR")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutPreselectionSR")

        # end
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutOS/?")
        self.assertEqual(cuts.GetEntries(), 2)
        self.assertEqual(cuts.At(0).GetName(), "CutBVeto")
        self.assertEqual(cuts.At(1).GetName(), "CutBReq")

    def test_get_matching_cuts_path_with_question_mark_non_existing(self):
        """
        Test getMatchingCuts when supplied with a cut path with a question
        mark, which does not match any cut.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutNonExisting/?/CutBVeto/CutPreselectionSR")
        self.assertEqual(cuts.GetEntries(), 0)

    def test_get_matching_cuts_path_with_asterisk(self):
        """
        Test getMatchingCuts when supplied with a cut path with an asterisk.
        """
        # beginning
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "*/CutPreselectionSR")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutPreselectionSR")

        # mid
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/*/CutPreselectionSR")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutPreselectionSR")

        # end
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutOS/CutBVeto/*")
        self.assertEqual(cuts.GetEntries(), 4)
        self.assertEqual(cuts.At(0).GetName(), "CutBVeto")
        self.assertEqual(cuts.At(1).GetName(), "CutPreselectionSR")
        self.assertEqual(cuts.At(2).GetName(), "CutVbfSR")
        self.assertEqual(cuts.At(3).GetName(), "CutBoostedSR")

    def test_get_matching_cuts_path_with_two_trailing_asterisk(self):
        """
        Test getMatchingCuts when supplied with a cut path with two trailing asterisk.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutVbfTopCR/*/*")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutVbfTopCR")

    def test_get_matching_cuts_path_with_optional_asterisk(self):
        """
        Test getMatchingCuts when supplied with a cut path containing an
        optional asterisk
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutPreselectionSR/*/CutVbfSR")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutVbfSR")

    def test_get_matching_cuts_path_with_asterisk_non_existing(self):
        """
        Test getMatchingCuts when supplied with a cut path with an asterisk,
        which does not match any cut.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutNonExisting/*/CutBVeto/CutPreselectionSR")
        self.assertEqual(cuts.GetEntries(), 0)

    def test_get_matching_cuts_path_with_asterisk_inner_name(self):
        """
        Test getMatchingCuts when supplied with a cut path with an asterisk in
        a cut name.
        """
        # mid
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutOS/CutB*/CutPreselection*R")
        self.assertEqual(cuts.GetEntries(), 2)
        self.assertEqual(cuts.At(0).GetName(), "CutPreselectionSR")
        self.assertEqual(cuts.At(1).GetName(), "CutPreselectionTopCR")

        # end
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutOS/CutB*")
        self.assertEqual(cuts.GetEntries(), 2)
        self.assertEqual(cuts.At(0).GetName(), "CutBVeto")
        self.assertEqual(cuts.At(1).GetName(), "CutBReq")

    def test_get_matching_cuts_path_with_asterisk_inner_name_non_existing(self):
        """
        Test getMatchingCuts when supplied with a cut path with an asterisk in
        a cut name, which does not match any cut.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutTrigger/CutNonExisting/CutOS/CutB*/CutNonExisting")
        self.assertEqual(cuts.GetEntries(), 0)

    def test_get_matching_cuts_astersik_only(self):
        """
        Test getMatchingCuts when supplied with a single asterisk.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "*")
        self.assertEqual(cuts.GetEntries(), 11)

    def test_get_matching_cuts_question_mark_only(self):
        """
        Test getMatchingCuts when supplied with a single question mark.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "?")
        self.assertEqual(cuts.GetEntries(), 11)

    def test_get_matching_cuts_two_astersik(self):
        """
        Test getMatchingCuts when supplied with a two asterisks.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "*/*")
        self.assertEqual(cuts.GetEntries(), 11)

    def test_get_matching_cuts_question_mark_asterisk(self):
        """
        Test getMatchingCuts when supplied with a question mark and an
        asterisk.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "?/*")
        self.assertEqual(cuts.GetEntries(), 11)

    def test_get_matching_cuts_two_question_mark(self):
        """
        Test getMatchingCuts when supplied with two question marks.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "?/?")
        # matches all except CutBase
        self.assertEqual(cuts.GetEntries(), 10)

    def test_get_matching_cuts_path_non_base(self):
        """
        Test getMatchingCuts when supplied with a cut path not starting at the
        base cut.
        """
        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutPreselection*/*SR")
        self.assertEqual(cuts.GetEntries(), 2)
        self.assertEqual(cuts.At(0).GetName(), "CutVbfSR")
        self.assertEqual(cuts.At(1).GetName(), "CutBoostedSR")

        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutPreselectionSR/CutVbf?R")
        self.assertEqual(cuts.GetEntries(), 1)
        self.assertEqual(cuts.At(0).GetName(), "CutVbfSR")

        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutOS/?")
        self.assertEqual(cuts.GetEntries(), 2)
        self.assertEqual(cuts.At(0).GetName(), "CutBVeto")
        self.assertEqual(cuts.At(1).GetName(), "CutBReq")

        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutOS/*")
        self.assertEqual(cuts.GetEntries(), 9)
        self.assertEqual(cuts.At(0).GetName(), "CutOS")
        self.assertEqual(cuts.At(1).GetName(), "CutBVeto")
        self.assertEqual(cuts.At(2).GetName(), "CutPreselectionSR")
        self.assertEqual(cuts.At(3).GetName(), "CutVbfSR")
        self.assertEqual(cuts.At(4).GetName(), "CutBoostedSR")
        self.assertEqual(cuts.At(5).GetName(), "CutBReq")
        self.assertEqual(cuts.At(6).GetName(), "CutPreselectionTopCR")
        self.assertEqual(cuts.At(7).GetName(), "CutVbfTopCR")
        self.assertEqual(cuts.At(8).GetName(), "CutBoostedTopCR")

        base = self.get_cut_hierarchy()
        cuts = TObjArray()
        base.getMatchingCuts(cuts, "CutOS/Cut*")
        self.assertEqual(cuts.GetEntries(), 2)
        self.assertEqual(cuts.At(0).GetName(), "CutBVeto")
        self.assertEqual(cuts.At(1).GetName(), "CutBReq")

    def test_add_analysis_job_singal_cut_name_asterisk_inner_name(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to the single cut
        specified by its name when using an asterisk in the name. (e.g.  "@CutNa*: histos;") 
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutPresel*SR")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)

    def test_add_analysis_job_singal_cut_name_question_mark_inner_name(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to the single cut
        specified by its name when using a question mark in the name. (e.g.
        "@CutNa*e: histos;") 
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutPreselection?R")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)

    def test_add_analysis_job_singal_cut_name_question_mark(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to the single cut
        specified by its name when using a question mark.
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutOS/?/CutPreselectionSR")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 1)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)

    def test_add_analysis_job_singal_cut_name_asterisk(self):
        """
        Test whether addAnalysisJob attaches an analsyis job to the single cut
        specified by its name when using a asterisk.
        """
        analysis_job = TQHistoMakerAnalysisJob()
        base = self.get_cut_hierarchy()
        base.addAnalysisJob(analysis_job, "CutOS/*/CutVbfSR")

        trigger = base.getCut("CutTrigger")
        OS = trigger.getCut("CutOS")
        BVeto = OS.getCut("CutBVeto")
        PreselectionSR = BVeto.getCut("CutPreselectionSR")
        VbfSR = PreselectionSR.getCut("CutVbfSR")
        BoostedSR = PreselectionSR.getCut("CutBoostedSR")
        BReq = OS.getCut("CutBReq")
        PreselectionTopCR = BReq.getCut("CutPreselectionTopCR")
        VbfTopCR = PreselectionTopCR.getCut("CutVbfTopCR")
        BoostedTopCR = PreselectionTopCR.getCut("CutBoostedTopCR")

        self.assertEqual(base.getNAnalysisJobs(), 0)
        self.assertEqual(trigger.getNAnalysisJobs(), 0)
        self.assertEqual(BVeto.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionSR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfSR.getNAnalysisJobs(), 1)
        self.assertEqual(BoostedSR.getNAnalysisJobs(), 0)
        self.assertEqual(PreselectionTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(VbfTopCR.getNAnalysisJobs(), 0)
        self.assertEqual(BoostedTopCR.getNAnalysisJobs(), 0)

