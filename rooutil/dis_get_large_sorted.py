#!/bin/env python

import dis_client
import sys
response =  dis_client.query(q=sys.argv[1], typ="basic", detail=False)
samples = response["response"]["payload"]
samples_nevents = []
for sample in samples:
    response = dis_client.query(q=sample, typ="basic", detail=False)
    nevent = response["response"]["payload"]["nevents"]
    samples_nevents.append((sample, nevent))
samples_nevents.sort(key=lambda x: x[1], reverse=True)
for sample, nevent in samples_nevents:
    print sample, nevent
