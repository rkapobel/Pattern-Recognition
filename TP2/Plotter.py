#!/usr/bin/python
import matplotlib.pyplot as plot
import os
import numpy as np

imageDirectory = os.path.dirname(os.path.realpath(__file__)) + "/Images/"

if not os.path.exists(imageDirectory):
    os.makedirs(imageDirectory)

markers = ["x", "o", "v", "s", "p", "1", "2", "3", "4", "8", "*", "h", "H", "+", "X", "D", "d", "|", "_", ".", ",", "^", "<", ">"]
colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

def plotErrorsByDegree(degrees, errors, imageName):
    plot.clf()
    updatePlotParams()
    plot.plot(degrees, errors[0], "b-o", label="Training")
    plot.plot(degrees, errors[1], "r-o", label="Test")
    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_title("$Degree$ $influence$")
    plot.axes().set_xlabel("$M$")
    plot.axes().set_ylabel("$E_{drms}$")
    plot.yscale("log", nonposy="clip")
    plot.savefig(os.path.join(imageDirectory, imageName + ".pdf"))

def plotErrorsByLogLambda(lambdas, errors, imageName):
    plot.clf()
    updatePlotParams()
    plot.plot(lambdas, errors[0], "b-o", label="Training")
    plot.plot(lambdas, errors[1], "r-o", label="Test")
    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_title("$\lambda$ $influence$")
    plot.axes().set_xlabel("$log(\lambda)$")
    plot.axes().set_ylabel("$E_{rms}$")
    plot.yscale("log", nonposy="clip")
    plot.savefig(os.path.join(imageDirectory, imageName + ".pdf"))
    
def plotOriginalVsEstimated(fOriginal, fEstimated, data, trngData, f1Name, degree, reg, imageName):
    plot.clf()
    updatePlotParams()
    plot.plot(data, [fOriginal(x) for x in data], "b-", label=f1Name)
    plot.plot(data, [fEstimated(x) for x in data], "r-", label="Estimated")
    plot.plot(trngData[0], trngData[1], "bo", label="Training data")
    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_title("$" + f1Name + "$ $vs$ $Polynom$ $degree$: $" + str(degree) + "$ $\lambda$: $" + str(reg) + "$")
    plot.axes().set_xlabel("$x$")
    plot.axes().set_ylabel("$y$")
    plot.ylim(-3.1, 3.1)
    plot.savefig(os.path.join(imageDirectory, imageName + ".pdf"))

def plotClasses(classes, classificated, imageName):
    plot.clf()
    updatePlotParams()
    
    for i in xrange(len(classes)):
        ci = classes[i]
        if len(ci) > 0:
            mark = colors[(i / len(markers)) % len(colors)] + markers[i % len(markers)]
            cx1, cx2 = zip(*ci)
            plot.plot(cx1, cx2, mark, label="Training class {0}".format(i))
    
    for i in xrange(len(classificated)):
        ci = classificated[i]
        if len(ci) > 0:
            mark = colors[((i / len(markers)) + 1) % len(colors)] + markers[i % len(markers)]
            cx1, cx2 = zip(*ci)
            plot.plot(cx1, cx2, mark, label="Test class {0}".format(i))
    
    plot.legend(loc="upper right", shadow=True)
    plot.savefig(os.path.join(imageDirectory, imageName + ".pdf"))

def plotCosts(epochs, costsPerClass, imageName):
    plot.clf()
    updatePlotParams()

    i = 0
    for costs in costsPerClass:
        mark = colors[((i / len(markers)) + 1) % len(colors)] + markers[i % len(markers)]
        plot.plot(epochs[0: len(costs)], costs, mark, label="Class {0}".format(i))
        i += 1

    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_xlabel("$Epochs$")
    plot.axes().set_ylabel("$Cost$")
    plot.savefig(os.path.join(imageDirectory, imageName + ".pdf"))

def plotConvergence(epochs, data, imageName, yTitle):
    plot.clf()
    updatePlotParams()
    plot.plot(epochs, data, 'ob', label=yTitle)
    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_xlabel("$Epochs$")
    plot.axes().set_ylabel("${0}$".format(yTitle))
    plot.savefig(os.path.join(imageDirectory, imageName + ".pdf"))

def updatePlotParams():
    params = {
        'legend.fontsize': 7,
        'legend.handlelength': 2
    }
    plot.rcParams.update(params)