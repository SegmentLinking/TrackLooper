#!/usr/bin/env python

import commands
import datetime
import time
from pprint import pprint
import json

def runtime2secs(rt):
    secs = 0
    days, intraday = runtime.split("+")
    hours, mins, secs = map(int,intraday.split(":"))
    secs += int(days)*3600*24 + hours*3600 + mins*60 + secs
    return secs

thisyear = datetime.datetime.now().year

stat, out = commands.getstatusoutput("condor_q -w")

# key is username and value is dict with key of jobids and values with the rest of the stuff
jobs = {"users": {}}
for line in out.splitlines()[4:-2]:
    line = line.strip()
    parts = line.split()
    jobid, user, daydate, timedate, runtime, status = parts[:6]
    datestr = daydate + " " + timedate
    executable = parts[8]
    args = " ".join(parts[9:])
    ts = int(time.mktime(datetime.datetime.strptime(datestr,"%m/%d %H:%M").replace(year=thisyear).timetuple()))
    runtime = runtime2secs(runtime)

    # slim the datatupler condor jobs a bit
    if args.split()[0].strip() == "pset_data.py":
        args = args.split()[-1].strip()


    job = {"ts": ts, "runtime": runtime, "status": status, "executable": executable, "args": args}
    if user not in jobs["users"]: jobs["users"][user] = {}
    jobs["users"][user][jobid] = job

jobs["info"] = {}
jobs["info"]["ts"] = int(time.mktime(datetime.datetime.now().timetuple()))
jobs["info"]["host"] = commands.getstatusoutput("hostname")[1]


print json.dumps(jobs)
# pprint(jobs)

# import pickle
# import gzip
# with gzip.open("test.pkl", "w") as fhout:
#     pickle.dump(jobs, fhout)

