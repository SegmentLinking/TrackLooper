import ROOT as r

f = r.TFile("/hadoop/cms/store/user/slava77/CMSSW_10_4_0_patch1-tkNtuple/pass-e072c1a/27411.0_TenMuExtendedE_0_200/trackingNtuple.root")
t = f.Get("trackingNtuple/tree")

for event in t:

    for isimhit, vecint in enumerate(event.simhit_hitIdx):

        if len(vecint) > 1:

            for typ, idx in zip(event.simhit_hitType[isimhit], event.simhit_hitIdx[isimhit]):

                if typ != 0:

                    print typ, event.ph2_x[idx], event.simhit_x[isimhit]
