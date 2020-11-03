
caps_lut = {
        "G": u"\u0262",
        "I": u"\u026A",
        "N": u"\u0274",
        "R": u"\u0280",
        "Y": u"\u028F",
        "B": u"\u0299",
        "H": u"\u029C",
        "L": u"\u029F",
        "A": u"\u1D00",
        "C": u"\u1D04",
        "D": u"\u1D05",
        "E": u"\u1D07",
        "J": u"\u1D0A",
        "K": u"\u1D0B",
        "M": u"\u1D0D",
        "O": u"\u1D0F",
        "P": u"\u1D18",
        "T": u"\u1D1B",
        "U": u"\u1D1C",
        "V": u"\u1D20",
        "W": u"\u1D21",
        "Z": u"\u1D22",
        "F": u"\uA730",
        "S": u"\uA731",
        }

col_lut = {
        "black"   : 0,
        "red"     : 1,
        "green"   : 2,
        "yellow"  : 3,
        "blue"    : 4,
        "magenta" : 5,
        "cyan"    : 6,
        "gray"    : 7,
        "white"   : 67,
        }


def get_colortext(text,fg=None,bg=None,modifiers=[],small_caps=False):
    buff = "\033["
    if small_caps:
        text = "".join([caps_lut.get(c.upper(),c.upper()) for c in text])
    for imod,mod in enumerate(modifiers):
        if mod == "bold": buff += ";1"
        elif mod == "dim": buff += ";2"
        elif mod == "underlined": buff += ";4"
        elif mod == "blink": buff += ";5"
        elif mod == "invert": buff += ";7"
        elif mod == "hidden": buff += ";8"
    for col,which in zip([fg,bg],["fg","bg"]):
        if type(col) in [tuple,list] and len(col) == 3:
            num = 38
            if which == "bg": num += 10
            buff += ";{};2;{};{};{}".format(num,col[0],col[1],col[2])
        elif col in col_lut.keys():
            code = col_lut[col] + (40 if which == "bg" else 30)
            buff += ";{}".format(code)
    if not fg and not bg and not modifiers:
        return text
    buff += "m"
    buff += text
    buff += "\033[0m"
    # print buff.replace("\033","")
    return buff

def print_info(text):
    print get_colortext(" info ",bg=(224,229,233),fg=(53,127,222),small_caps=True)+"  "+text

def print_debug(text):
    print get_colortext(" debug ",bg=(235, 231, 111),fg=(149, 3, 25),small_caps=True)+" "+text

def print_error(text):
    print get_colortext(" error ",bg=(186, 35, 55), fg=(242, 207, 14),small_caps=True)+" "+text

def print_note(text):
    print get_colortext(" note ",bg="black", fg="white",small_caps=True)+" "+text

if __name__ == "__main__":
    print_info("my test")
    print_debug("my test")
    print_error("my test")
    print_info("my test")
    print_note("my test")

    # print get_colortext(" info ",bg=(224,229,233),fg=(53,127,222),small_caps=True)
    # print get_colortext(" debug ",bg=(235, 231, 111),fg=(149, 3, 25),small_caps=True)
    # print get_colortext(" error ",bg=(186, 35, 55), fg=(242, 207, 14),small_caps=True)


