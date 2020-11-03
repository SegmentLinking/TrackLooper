#!/usr/bin/env python

import json
import os
import sys
import commands
import time

def main():
    key = "AIzaSyBvJEPLij7Lr-XNU58sIhcjmGgcgjKa4AU"
    do_test = False

    if do_test:
        output = """
                                    SSID BSSID             RSSI CHANNEL HT CC SECURITY (auth/unicast/group)
                    DIRECTV_WVB_91827C0E 8c:43:8c:43:2d:2d -88  44,+1   Y  -- WPA2(PSK/AES/AES)
           DIRECT-4F-HP ENVY 4520 series 70:73:70:73:50:50 -68  11      Y  -- WPA2(PSK/AES/AES)
                                2WIRE424 94:4d:94:4d:81:81 -19  1       N  US WPA(PSK/AES,TKIP/TKIP) WPA2(PSK/AES,TKIP/TKIP)
                                """
    else:
        itry = 0
        output = ""
        while itry < 3 and not output.strip():
            if itry > 0:
                print ">>> Didn't get anything from AirPort command, trying again in 1 second"
                time.sleep(3)
            status, output = commands.getstatusoutput("/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s")
            itry += 1
        


    points = []
    for line in output.splitlines():
        mac = None
        mac_idx = None
        name = ""
        strength = -999
        parts = line.split()
        for iword,word in enumerate(parts):
            if word.count(":") == 5:
                mac = word
                mac_idx = iword
        if mac_idx: 
            name = " ".join(parts[:mac_idx]).strip()
            strength = int(float(parts[mac_idx+1]))

        if mac and strength > -998:
            print ">>> Found access point %s (%s) with strength %i" % (name,mac,strength)
            points.append( [mac,strength] )

    if not points:
        print ">>> Found 0 wifi networks, try again later"
        sys.exit()

    d_info = {}
    d_info["considerIp"] = "false"
    d_info["wifiAccessPoints"] = []
    for mac, strength in points:
        d_info["wifiAccessPoints"].append({"macAddress": mac, "signalStrength": strength})

    with open("temp.json", "w") as fhout:
        fhout.write(json.dumps(d_info, indent=2))

    if do_test:
        output = """
        {
            "location": {
                "lat": 33.8799218,
                "lng": -93.29
            },
            "accuracy": 17.0
        }
        """
    else:
        status, output = commands.getstatusoutput(""" curl -s -d @temp.json -H "Content-Type: application/json" -i "https://www.googleapis.com/geolocation/v1/geolocate?key=%s" | grep "{" -A 50 """ % key)
        os.system("rm -f temp.json")


    try:
        lat = json.loads(output)["location"]["lat"]
        lng = json.loads(output)["location"]["lng"]
        acc = json.loads(output)["accuracy"]
        print
        print "lat,long: %f,%f\naccuracy: %f" % (lat, lng, acc)
        print "url: https://www.google.com/maps/place/%f,%f" % (lat,lng)
    except:
        print ">>> Couldn't parse output of google API query. Output was:"
        print output

if __name__ == "__main__":
    main()
