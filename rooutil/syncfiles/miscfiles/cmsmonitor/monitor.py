# requires tesseract
# On mac: brew install tesseract
#         brew link tesseract
import pytesseract as pt
import os,sys,commands
from PIL import Image
import datetime, time, json, re

sys.path.append("/usr/local/bin/tesseract")

def cleanText(txt):
    newtxt = []
    for line in txt.split("\n"):
        if(len(line.strip())) < 1: continue
        newtxt.append(line.strip())
    return newtxt

# setup: get images, crop them
os.system("curl -o lhc.png https://vistar-capture.web.cern.ch/vistar-capture/lhc1.png")
os.system("curl -o cms.png https://cmspage1.web.cern.ch/cmspage1/data/page1.png")

imgCMS = Image.open("cms.png")
imgLHC = Image.open("lhc.png")
imgCMStop = imgCMS.crop((0,0,800,30))
imgLHCtop = imgLHC.crop((0,0,1024,90))
imgCMSbfield = imgCMS.crop((710,433,782,446))
(width, height) = imgCMSbfield.size
scale = 4
imgCMSbfield = imgCMSbfield.resize((scale*width,scale*height), Image.BICUBIC)

# CMS subsystems
dy = (583.0-374)/13.0
systemsall = ["CSC","DT","ECAL","ES","HCAL","PIXEL","RPC","TRACKER","CASTOR","TRG","DAQ","DQM","SCAL","HF"]
systemsin = []
systemson = []
systemsgood = True
rgbimg = imgCMS.convert("RGB")
for i in range(14):
    coords = (498,374+i*dy)
    rgb = rgbimg.getpixel(coords)
    if(250 > rgb[1] > 240): systemsin.append(systemsall[i])
for i in range(14):
    coords = (556,374+i*dy)
    rgb = rgbimg.getpixel(coords)
    if(250 > rgb[1] > 240): systemson.append(systemsall[i])

numSystemsgood = 0
for system in ["CSC","DT","ECAL","ES","HCAL","PIXEL","RPC","TRACKER","TRG","DAQ","DQM","SCAL","HF"]:
    if system not in systemsin:
        systemsgood = False
    else:
        numSystemsgood += 1
for system in ["CSC","DT","ECAL","ES","HCAL","PIXEL","RPC","TRACKER"]:
    if system not in systemson:
        systemsgood = False
    else:
        numSystemsgood += 1

print systemsin, systemson, systemsgood

# Magnetic field
txt = pt.image_to_string(imgCMSbfield)
bfield = -0.001
try:
    for line in cleanText(txt):
        if "[T]" not in line: continue
        bfield = float(line.split(" ")[-1])
except: pass
# if we couldn't find it using [T], then regex for #.###
try:
    s = re.search("[0-9]\.[0-9][0-9][0-9]"," ".join(cleanText(txt)))
    if s and bfield < 0.0: bfield = float(s.group())
except: pass
if bfield < 0: print "couldn't get bfield from:", txt


# Run number
txt = pt.image_to_string(imgCMStop)
run = -1
try:
    clean = ''.join(i for i in cleanText(txt)[0] if (i.isdigit() or i==' '))
    for item in clean.split():
        try:
            maybeRun = int(item)
            if 200000 < maybeRun < 600000:
                run = maybeRun
                break
        except:
            pass
except: pass
if run < 0: print "couldn't get run from:", txt
print "run",run

txt = pt.image_to_string(imgLHCtop)

# Fill
fill = -1
try:
    fill = int(cleanText(txt)[0].split(":")[1].strip().split()[0])
except: pass
if fill < 0: print "couldn't get fill from:", txt
print "fill",fill

# Energy
energy = -1
try:
    energy = int(cleanText(txt)[0].split(":")[2].strip().split()[0])
except: pass
if energy < 0: print "couldn't get energy from:", txt
print "energy", energy

# Beam status
beam = ""
try:
    beam = cleanText(txt)[1].split(":")[1].strip().lower()
except: pass
if beam == "": print "couldn't get beam from:", txt
print "beam",beam

# Timestamp from monitor picture
timestamp, timestampparsed = "", 0
try:
    timestamp = " ".join(cleanText(txt)[0].split()[-2:]).strip()
    # 15-08-15 07:49:12
    timestampparsed = int(time.mktime(datetime.datetime.strptime(timestamp, "%d-%m-%y %H:%M:%S").timetuple()))
except: pass
if timestampparsed == 0: print "couldn't get timestamp from:", txt
print "timestamp",timestamp


info = {}
info["bfield"] = bfield
info["systemsin"] = systemsin
info["systemson"] = systemson
info["run"] = run
info["fill"] = fill
info["energy"] = energy
info["beam"] = beam
info["realtime"] = int((datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%s"))
info["timestamp"] = timestamp
info["timestampparsed"] = timestampparsed

info["good"] = {}
info["good"]["beamsgood"] = "stable" in beam
info["good"]["systemsgood"] = systemsgood
info["good"]["bfieldgood"] = bfield > 3.7
info["good"]["energygood"] = energy > 6390
info["good"]["realtime"] = int((datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%s"))
info["good"]["timestampparsed"] = timestampparsed



fractionalSystemsGood = 1.0*numSystemsgood/21
fractionalEnergy = 1.0*energy/6400
fractionalBfield = 1.0*bfield/3.8

# 3 points for full energy, 5 for full B, 5 for full systems, 2 for stable beams
scorefine = 3.0*fractionalEnergy + 5.0*fractionalBfield + 5.0*fractionalSystemsGood + 2.0*("stable" in beam)
scoregrainy = 5*systemsgood + 5*(bfield>3.7) + 5*("stable" in beam)
info["score"] = {}
info["score"]["scorefine"] = scorefine
info["score"]["scoregrainy"] = scoregrainy



print json.dumps(info, indent=4)
out = open("monitor.json","w")
out.write(json.dumps(info, indent=4))
out.close()

good = info["good"]
realtime = good["realtime"]
timestampparsed = good["timestampparsed"]
energygood = good["energygood"]
beamsgood = good["beamsgood"]
bfieldgood = good["bfieldgood"]
systemsgood = good["systemsgood"]
out = open("data.txt","a")
out.write("%i %i %i %i %i %i %i %.1f\n" % (realtime,timestampparsed,energygood,beamsgood,bfieldgood,systemsgood,scoregrainy,scorefine) )
out.close()

