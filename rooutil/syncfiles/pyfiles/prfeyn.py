import os
import ROOT as r

c1 = None

class Label(object):
    def __init__(self,text="",x1=0.0,y1=0.0,offsetx=0,offsety=0,textsize=0.08,textalign=22,roman=False):
        self.x1 = x1
        self.y1 = y1
        self.text = text
        self.offsetx = offsetx
        self.offsety = offsety
        self.textsize = textsize
        self.textalign = textalign
        self.roman = roman

    def set_location(self,x,y):
        self.x1 = x
        self.y1 = y

    def transform_text(self,text):
        # ROOT needs one of these characters to put in a $ and go into mathmode
        # otherwise we do it explicitly
        if self.roman or any([x in text for x in "#{}^"]): return text
        text = "${0}$".format(text)
        return text

    def draw(self):
        if not self.text: return
        t = r.TLatex()
        t.SetTextAlign(self.textalign)
        t.SetTextSize(self.textsize)
        t.DrawLatex(self.x1+self.offsetx,self.y1+self.offsety,self.transform_text(self.text))


class Vertex(object):
    def __init__(self, x1,y1, label=Label(), autolabel=True):
        self.x1 = x1
        self.y1 = y1
        self.label = label

        if autolabel:
            self.label.set_location(self.x1,self.y1)

    def draw(self):
        self.label.draw()


class Propagator(object):
    def __init__(self,  v1,v2, typ="line", label=Label(), autolabel=True, \
            linewidth=1, linecolor=r.kBlack, fliparrow=False,noarrow=False):
        self.v1 = v1
        self.v2 = v2
        self.typ = typ
        self.label = label
        self.linewidth = linewidth
        self.linecolor = linecolor
        self.fliparrow = fliparrow
        self.noarrow = noarrow

        if autolabel:
            self.label.set_location(
                    0.5*(self.v1.x1+self.v2.x1),
                    0.5*(self.v1.y1+self.v2.y1)
            )


    def draw(self, _nodelete=[]):

        prop1, prop2 = None, None
        if self.typ == "line":
            prop1 = r.TLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
        if self.typ == "dashedline":
            prop1 = r.TLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
            prop1.SetLineStyle(7)
        if self.typ == "dottedline":
            prop1 = r.TLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
            prop1.SetLineStyle(3)
        elif self.typ == "curlyline":
            prop1 = r.TCurlyLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
            prop1.SetWaveLength(prop1.GetWaveLength()*1.4)
            prop1.SetAmplitude(prop1.GetAmplitude()*1.4)
        elif self.typ == "wavyline":
            prop1 = r.TCurlyLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
            prop1.SetWavy()
            prop1.SetWaveLength(prop1.GetWaveLength()*1.4)
            prop1.SetAmplitude(prop1.GetAmplitude()*1.4)
        elif self.typ == "wavystraightline":
            prop1 = r.TCurlyLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
            prop1.SetWavy()
            prop1.SetWaveLength(prop1.GetWaveLength()*1.4)
            prop1.SetAmplitude(prop1.GetAmplitude()*1.4)
            prop2 = r.TLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
        elif self.typ == "curlystraightline":
            prop1 = r.TCurlyLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)
            prop1.SetWaveLength(prop1.GetWaveLength()*1.4)
            prop1.SetAmplitude(prop1.GetAmplitude()*1.4)
            prop2 = r.TLine(self.v1.x1, self.v1.y1, self.v2.x1, self.v2.y1)


        prop1.SetLineColor(self.linecolor)
        prop1.SetLineWidth(self.linewidth)

        if prop2:
            prop2.SetLineColor(self.linecolor)
            prop2.SetLineWidth(self.linewidth)

        if prop1: prop1.Draw()
        if prop2: prop2.Draw()

        # need this or else pyroot deletes the object and we don't see it anymore :(
        if prop1: _nodelete.append(prop1)
        if prop2: _nodelete.append(prop2)

        if self.typ in ["line", "dashedline"] and not self.noarrow:
            c1 = self.v1.x1,self.v1.y1
            c2 = self.v2.x1,self.v2.y1
            if self.fliparrow: c1, c2 = c2, c1
            mult = 0.57
            awidth = 0.025
            a1 = r.TArrow(c1[0],c1[1],(1.-mult)*c1[0]+mult*c2[0],(1.-mult)*c1[1]+mult*c2[1], awidth,"|>")
            a1.SetLineWidth(0)
            a1.SetFillColor(self.linecolor)
            a1.SetAngle(40)
            a1.Draw()
            _nodelete.append(a1)

        self.v1.draw()
        self.v2.draw()
        self.label.draw()

def draw_grid(_nodelete=[]):
    for i in range(10+1):
        xline = r.TLine(10*i,0,10*i,100)
        yline = r.TLine(0,10*i,100,10*i)
        xline.SetLineColor(r.kGray)
        yline.SetLineColor(r.kGray)
        xlab = r.TLatex(10*i,0,str(10*i))
        ylab = r.TLatex(0,10*i,str(10*i))
        xlab.SetTextAlign(23)
        ylab.SetTextAlign(32)
        xlab.SetTextColor(r.kGray+2)
        ylab.SetTextColor(r.kGray+2)
        xline.Draw()
        yline.Draw()
        xlab.Draw()
        ylab.Draw()
        _nodelete.append(xline)
        _nodelete.append(yline)
        _nodelete.append(xlab)
        _nodelete.append(ylab)

def init_diagram():
    global c1
    c1 = r.TCanvas("c1", "A canvas", 10,10, 600, 600);
    c1.Range(0, 0, 100, 100);

def save_diagram(fname):
    global c1

    fname = fname.replace(".pdf",".tex")
    c1.SaveAs(fname)
    with open("tmp.tex", "w") as fhout:
        with open(fname,"r") as fhin:
            fhout.write("\\documentclass{article}\n")
            fhout.write("\\usepackage{tikz}\n")
            fhout.write("\\usetikzlibrary{patterns}\n")
            fhout.write("\\usetikzlibrary{plotmarks}\n")
            fhout.write("\\begin{document}\n")
            fhout.write("\\pagenumbering{gobble}\n")
            fhout.write("\\par\n")
            fhout.write("\\begin{figure}[htbp]\n")
            fhout.write("\\scalebox{0.7}{\n")
            for line in fhin:
                fhout.write(line)
            fhout.write("}\n")
            fhout.write("\\end{figure}\n")
            fhout.write("\\end{document}\n")
    os.system("mv tmp.tex {0}".format(fname))

    tex_to_pdf(fname)

    fname = fname.replace(".tex",".pdf")
    crop_pdf(fname)
    web(fname)

def tex_to_pdf(fname):
    os.system("pdflatex -interaction=nonstopmode -q {0} >& /dev/null ".format(fname))

def crop_pdf(fname):
    os.system("pdfcrop --margins 4 {0} {0} >& /dev/null".format(fname))

def web(fname):
    os.system("web {0}".format(fname))

if __name__ == "__main__":

    ### Draw e+e- --> qqbar
    
    vep = Vertex(20,35, Label("e^{+}",offsetx=-2,offsety=-2))
    vem = Vertex(20,65, Label("e^{-}",offsetx=-2,offsety=2))
    veebl = Vertex(35,50)
    veebr = Vertex(65,50)
    vqp = Vertex(80,35, Label("q", offsetx=2,offsety=-2))
    vqm = Vertex(80,65, Label("#bar{q}", offsetx=2,offsety=2))

    pep = Propagator(vep,veebl,typ="line",fliparrow=True)
    pem = Propagator(vem,veebl,typ="line")
    pboson = Propagator(veebl,veebr,typ="wavyline", label=Label("Z^{0}/#gamma",offsety=6), linewidth=2, linecolor=r.kBlue+2)

    pqp = Propagator(vqp,veebr,typ="line")
    pqm = Propagator(vqm,veebr,typ="line",fliparrow=True)

    vgluonl = Vertex(65+7.5,50+7.5)
    vgluonr = Vertex(65+7.5+7.5+4,50-4, Label("g", offsetx=2,offsety=2))

    pglu = Propagator(vgluonl, vgluonr, typ="curlyline")

    lab = Label("e^+e^-\\rightarrow q\\bar{q}g", 50,30,  textsize=0.07)

    init_diagram()
    draw_grid()

    pep.draw()
    pem.draw()
    pboson.draw()
    pqp.draw()
    pqm.draw()
    pglu.draw()
    lab.draw()

    save_diagram("eetoqq.pdf")

    ### gg->ttH/A

    vg1 = Vertex(20,60, Label("g",offsetx=-2))
    vg2 = Vertex(20,30, Label("g",offsetx=-2))
    vg1t = Vertex(45,60)
    vg2t = Vertex(45,30)
    vg1t_right = Vertex(70,60, Label("t",offsetx=2))
    vg2t_right = Vertex(70,30, Label("t",offsetx=2))
    vtth = Vertex(45,45)
    vh_right = Vertex(70,45, Label("H(A)",offsetx=5))
    lw = 2
    pg1 = Propagator(vg1,vg1t,typ="curlyline",linewidth=lw)
    pg2 = Propagator(vg2,vg2t,typ="curlyline",linewidth=lw)
    ptth_midtop = Propagator(vtth,vg1t,typ="line",label=Label("t",offsetx=-2),linecolor=r.kRed-4,linewidth=lw)
    ptth_midbot = Propagator(vtth,vg2t,typ="line",label=Label("t",offsetx=-2),linecolor=r.kRed-4,linewidth=lw, fliparrow=True)
    ptth_top = Propagator(vg1t,vg1t_right,typ="line",linecolor=r.kRed-4,linewidth=lw)
    ptth_bot = Propagator(vg2t,vg2t_right,typ="line",linecolor=r.kRed-4,linewidth=lw, fliparrow=True)
    ptth_mid = Propagator(vtth,vh_right,typ="dashedline",linecolor=r.kAzure-6,linewidth=lw)

    init_diagram()
    draw_grid()

    pg1.draw()
    pg2.draw()
    ptth_midtop.draw()
    ptth_midbot.draw()
    ptth_top.draw()
    ptth_bot.draw()
    ptth_mid.draw()

    save_diagram("tth.pdf")
