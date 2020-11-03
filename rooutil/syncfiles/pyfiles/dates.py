from datetime import date
import random, os, time
    
isCorrect = False

while(not isCorrect):
    t1 = time.time()
    thisYear = random.randint(0, 2)
    
    onlyThisYear = False 
    # 33% chance that it'll be from this year (2014)
    if(thisYear == 2 or onlyThisYear): randDay = random.randint(735234,735234+364) # 2014
    else: randDay = random.randint(694000,735234+364) # 1900-2014
    

    d = date.fromordinal(randDay)
    
    width = 79
    dateStr = (d.strftime("%B") + d.strftime(" %d, ") + d.strftime("%Y"))
    left = (width-len(dateStr)-4)/2 - 2
    right = width - (left + 2 + len(dateStr) + 2) - 4
    
    print "#" * (width)
    print "#" * (width)
    print "#" * (left), "  ",
    print dateStr,
    print "  ", "#" * (right)
    print "#" * (width)
    print "#" * (width)
    
    tips = """
    - Doomsday is January 3 (non-leap), 4 (leap)
    - Doomsday is February 28 or 29 (leap)
    - Doomsday is March "0"th
    - Doomsday is Nth of even months (except Feb)
    - Doomsday for odd months: I work 9-5 at a 7-11
    
    1.  Let T be the year's last two digits
    2.  If T is odd, add 11
    3.  Let T = T/2
    4.  If T is odd, add 11
    5.  Let T = 7 - T%7
    
    1900s - (3) Wednesday (We-in-dis-day)
    2000s - (2) Tuesday (Today)
    
    2014 - (5) Friday

    A. Get base anchor day for CENTURY
    B. Add T days to it
    C. Use Doomsdays to get final day

    """
    
    print tips
    print d
    inputDay = raw_input("Enter day: ")
    if(inputDay.capitalize() == d.strftime("%A")):
        isCorrect = True
        
        print "Correct!"
        print "Took you %d seconds to answer." % (time.time() - t1)
        time.sleep(1)
        os.system("exit")
    else:
        isCorrect = False
        print "Wrong. Correct answer:", d.strftime("%A")
        time.sleep(1)
    print "\n"

