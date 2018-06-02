#!/usr/bin/python
import matplotlib.pyplot as plot
import os

myPath = os.path.dirname(os.path.realpath(__file__))

markers = ["x", "o", "v", ".", ",", "^", "<", ">", "s", "p", "P", "*", "h", "H", "+", "X", "D", "d", "|", "_", "1", "2", "3", "4", "8"]
colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

def plotErrorsByDegree(degrees, errors, imageName):
    plot.clf()
    plot.plot(degrees, errors[0], "b-o", label="Training")
    plot.plot(degrees, errors[1], "r-o", label="Test")
    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_title("$Degree$ $influence$")
    plot.axes().set_xlabel("$M$")
    plot.axes().set_ylabel("$E_{drms}$")
    plot.yscale("log", nonposy="clip")
    plot.savefig(os.path.join(myPath, imageName + ".pdf"))

def plotErrorsByLogLambda(lambdas, errors, imageName):
    plot.clf()
    plot.plot(lambdas, errors[0], "b-o", label="Training")
    plot.plot(lambdas, errors[1], "r-o", label="Test")
    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_title("$\lambda$ $influence$")
    plot.axes().set_xlabel("$log(\lambda)$")
    plot.axes().set_ylabel("$E_{rms}$")
    plot.yscale("log", nonposy="clip")
    plot.savefig(os.path.join(myPath, imageName + ".pdf"))
    
def plotOriginalVsEstimated(fOriginal, fEstimated, data, trngData, f1Name, degree, reg, imageName):
    plot.clf()
    plot.plot(data, [fOriginal(x) for x in data], "b-", label=f1Name)
    plot.plot(data, [fEstimated(x) for x in data], "r-", label="Estimated")
    plot.plot(trngData[0], trngData[1], "bo", label="Training data")
    plot.legend(loc="upper right", shadow=True)
    plot.axes().set_title("$" + f1Name + "$ $vs$ $Polynom$ $degree$: $" + str(degree) + "$ $\lambda$: $" + str(reg) + "$")
    plot.axes().set_xlabel("$x$")
    plot.axes().set_ylabel("$y$")
    plot.ylim(-3.1, 3.1)
    plot.savefig(os.path.join(myPath, imageName + ".pdf"))

def plotClasses(classes, classificated, imageName):
    plot.clf()
    for i in xrange(len(classes)):
        cl = classes[i]
        mark = colors[(i / len(markers)) % len(colors)] + markers[i % len(markers)]
        [plot.plot(point[0], point[1], mark) for point in cl]

    for i in xrange(len(classificated)):
        clPoints = classificated[i]
        for point in clPoints:
            mark = colors[((i / len(markers)) + 1) % len(colors)] + markers[i % len(markers)]
            plot.plot(point[0], point[1], mark)
    
    plot.savefig(os.path.join(myPath, imageName + ".pdf"))
